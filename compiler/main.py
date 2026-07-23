from .parser import program
from .dsl_ast import Program
from .printer import Printer

def compile_dsl(source: str, streams_per_gpu: int) -> str:
    printer = Printer(streams_per_gpu)
    ast = program.parse(source)
    return printer.print_program(ast)
