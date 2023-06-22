from dataclasses import dataclass, field
from io import StringIO
import random
import textwrap
import os
import sys
import traceback
from collections import defaultdict

from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
    RegionBlock,
)

from .mock_asm import ProgramGen, parse, VM, simulate_scfg, to_scfg



def test_mock_asm():
    asm = textwrap.dedent(
        """
            print Start
            goto A
        label A
            print A
            ctr 10
            brctr A B
        label B
            print B
    """
    )

    instlist = parse(asm)
    assert instlist[0].operands.text == "Start"
    assert instlist[1].operands.jump_target == 2
    assert instlist[2].operands.text == "A"
    assert instlist[3].operands.counter == 10
    assert instlist[4].operands.true_target == 2
    assert instlist[4].operands.false_target == 5
    assert instlist[5].operands.text == "B"

    with StringIO() as buf:
        VM(buf).run(instlist)
        got = buf.getvalue().split()

    expected = ["Start", *(["A"] * 10), "B"]
    assert got == expected


def test_double_exchange_loop():
    asm = textwrap.dedent(
        """
            print Start
       label A
            print A
            ctr 4
            brctr B Exit
        label B
            print B
            ctr 5
            brctr A Exit
        label Exit
            print Exit
    """
    )
    instlist = parse(asm)
    with StringIO() as buf:
        VM(buf).run(instlist)
        got = buf.getvalue().split()

    expected = ["Start", *(["A", "B"] * 3), "A", "Exit"]
    assert got == expected


def test_program_gen():
    rng = random.Random(123)
    pg = ProgramGen(rng)
    ct_term = 0
    total = 10000
    for i in range(total):
        print(str(i).center(80, "="))
        asm = pg.generate_program()

        instlist = parse(asm)
        with StringIO() as buf:
            terminated = VM(buf).run(instlist, max_step=1000)
            got = buf.getvalue().split()
            if terminated:
                print(asm)
                print(got)
                ct_term += 1
    print("terminated", ct_term, "total", total)



def compare_simulated_scfg(asm):
    instlist = parse(asm)
    scfg = to_scfg(instlist)

    with StringIO() as buf:
        terminated = VM(buf).run(instlist, max_step=1000)
        assert terminated
        expect = buf.getvalue()
    print("EXPECT".center(80, "="))
    print(expect)

    got = simulate_scfg(scfg)
    assert got == expect   # failed simluation

    return scfg


def ensure_contains_region(scfg: SCFG, loop: int, branch: int):
    def recurse_find_regions(bbmap: SCFG):
        for blk in bbmap.graph.values():
            if isinstance(blk, RegionBlock):
                yield blk
                yield from recurse_find_regions(blk.subregion)

    regions = list(recurse_find_regions(scfg))
    count_loop = 0
    count_branch = 0
    for reg in regions:
        if reg.kind == "loop":
            count_loop += 1
        elif reg.kind == "branch":
            count_branch += 1

    assert loop == count_loop
    assert branch == count_branch


def test_mock_scfg_loop():
    asm = textwrap.dedent(
        """
            print Start
            goto A
        label A
            print A
            ctr 10
            brctr A B
        label B
            print B
    """
    )
    scfg = compare_simulated_scfg(asm)
    ensure_contains_region(scfg, loop=1, branch=0)


def test_mock_scfg_head_cycle():
    # Must manually enforce the the entry block only has no predecessor
    asm = textwrap.dedent(
        """
            print Start
            goto S
        label S
            print S
            goto A
        label A
            print A
            ctr 10
            brctr S B
        label B
            print B
    """
    )
    scfg = compare_simulated_scfg(asm)
    ensure_contains_region(scfg, loop=1, branch=0)


def test_mock_scfg_diamond():
    asm = textwrap.dedent(
        """
            print Start
            ctr 1
            brctr A B
        label A
            print A
            goto C
        label B
            print B
            goto C
        label C
            print C
    """
    )
    scfg = compare_simulated_scfg(asm)
    ensure_contains_region(scfg, loop=0, branch=2)


def test_mock_scfg_double_exchange_loop():
    asm = textwrap.dedent(
        """
            print Start
            goto A
       label A
            print A
            ctr 4
            brctr B Exit
        label B
            print B
            ctr 5
            brctr A Exit
        label Exit
            print Exit
    """
    )
    scfg = compare_simulated_scfg(asm)
    # branch count may be more once branch restructuring is fixed
    ensure_contains_region(scfg, loop=1, branch=4)


def test_mock_scfg_doubly_loop():
    asm = textwrap.dedent(
        """
            print Entry
            goto Head
        label Head
            print Head
            goto Midloop
        label Midloop
            print Midloop
            ctr 2
            brctr Head Tail
        label Tail
            print Tail
            ctr 3
            brctr Midloop Exit
        label Exit
            print Exit
    """
    )
    scfg = compare_simulated_scfg(asm)
    # Note: branch number if not correct
    ensure_contains_region(scfg, loop=2, branch=6)


def run_fuzzer(seed):
    rng = random.Random(seed)
    pg = ProgramGen(rng)
    asm = pg.generate_program()

    print(seed)
    instlist = parse(asm)
    with StringIO() as buf:
        terminated = VM(buf).run(instlist, max_step=1000)
        got = buf.getvalue().split()
        if terminated:
            print(f"seed={seed}".center(80, "="))
            print(asm)
            print(got)

            compare_simulated_scfg(asm)
            return True


KNOWN_ERRORS = [
    # infinite loop caused by invalid control variable
    # failing seeds: [9, 122, 342, 382, 422, 571, 606, 659, 693, 715, 850, 868,
    #                 927, 943, 961]
    "MaxStepError: step > max_step",

    # non-infinite loop failures probably caused by invalid control variable
    # failing seeds: [60, 194, 312, 352, 461, 473, 595, 602, 720, 803, 831]
    "assert got == expect",

    # https://github.com/numba/numba-rvsdg/issues/44
    # failing seeds: [0, 7, 29, 44, 74, 88, 155, 253, 258, 295, 360, 401, 406,
    #                 530, 539, 554, 577, 629, 635, 638, 695, 773, 819, 829,
    #                 857, 866, 917]
    "next(iter(exit_blocks))",

    # https://github.com/numba/numba-rvsdg/issues/48
    # failing seeds: [58, 153, 262, 275, 476, 607, 724, 742, 763, 824, 832, 928,
    #                 930, 934, 945, 955, 984, 999]
    "assert len(diff) == 1",
]
# Run below to trigger specific errors:
# > python -m numba_rvsdg.tests.test_mock_asm <seed>

def check_against_known_error():
    # extract error message
    with StringIO() as fout:
        traceback.print_exc(file=fout)
        msg = fout.getvalue()

    # find associated group and return the group index
    for idx, err in enumerate(KNOWN_ERRORS):
        if err in msg:
            print("EXCEPTION".center(80, '<'))
            print(msg)
            print(">" * 80)
            return idx
    # otherwise return None
    return None


def test_mock_scfg_fuzzer(total=1000):
    # tested up to total=100000
    ct_term = 0

    known_failures = defaultdict(list)
    unknown_failures = set()

    for i in range(total):
        try:
            if run_fuzzer(i):
                ct_term += 1
        except Exception:
            print("Failed case:", i)
            err_idx = check_against_known_error()
            if err_idx is not None:
                known_failures[err_idx].append(i)
            else:
                unknown_failures.add(i)
        else:
            print("ok", i)
    print("terminated", ct_term, "total", total)
    print("known_failures:")
    for err_group, cases in sorted(known_failures.items()):
        print(f"  {err_group}: {cases}")
    print("unknown_failures: ", sorted(unknown_failures))
    assert not unknown_failures


"""
# Interesting but failing cases

def test_mock_scfg_fuzzer_case0():
    # https://github.com/numba/numba-rvsdg/issues/44
    run_fuzzer(seed=0)

def test_mock_scfg_fuzzer_case9():
    # invalid control variable causes infinite loop
    run_fuzzer(seed=9)

def test_mock_scfg_fuzzer_case60():
    # probably invalid control variable
    run_fuzzer(seed=60)

def test_mock_scfg_fuzzer_case153():
    # https://github.com/numba/numba-rvsdg/issues/48
    run_fuzzer(seed=153)
"""


if __name__ == "__main__":
    seed = int(sys.argv[1])
    run_fuzzer(seed=seed)
