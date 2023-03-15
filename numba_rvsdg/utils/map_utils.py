import dis

def bcmap_from_bytecode(bc: dis.Bytecode):
    return {inst.offset: inst for inst in bc}
