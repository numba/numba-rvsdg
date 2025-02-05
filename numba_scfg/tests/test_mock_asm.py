# mypy: ignore-errors

from io import StringIO
import random
import textwrap

from mock_asm import ProgramGen, parse, VM


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
