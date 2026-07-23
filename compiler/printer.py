from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from abc import ABC 
from parsy import from_enum, regex, seq, string, decimal_digit
from functools import partial
from .dsl_ast import *

def ncclString(command: Command):
    match command:
        case AllReduce():
            return "ncclAllReduce"
        case Broadcast():
            return "ncclBroadcast"
        case Gather():
            return "ncclGather"
        case Scatter():
            return "ncclScatter"
        case AllGather():
            return "ncclAllGather"
        case ReduceScatter():
            return "ncclReduceScatter"
        case Reduce():
            return "ncclReduce"
        case AllToAll():
            return "ncclAlltoAll"
        case Send():
            return "ncclSend"
        case Recv():
            return "ncclRecv"

class Printer:
    def __init__(self, streams_per_gpu: int):
        self.current_device = -1
        self.streams_per_gpu = streams_per_gpu

    def _stream_index(self, device: int, stream: int) -> str:
        return f"getStream({device}, {stream})"

    def print_program(self, program: Program) -> str:
        num_threads = len(program.threads)
        
        output = [
            f"std::barrier sync_point({num_threads});",
            "std::vector<std::jthread> threads;"
        ]
        
        for thread in program.threads:
            output.append(self.print_thread(thread))
            
        return "\n".join(output)

    def print_thread(self, thread: Thread) -> str:
        self.current_device = -1 
        
        output = [
            "threads.emplace_back([&]() {",
            "    sync_point.arrive_and_wait();"
        ]
        
        for node in thread.commands:
            cmd_out = self.print_command(node)
            
            indented = "\n".join(f"    {line}" for line in cmd_out.split("\n"))
            output.append(indented)
            
        output.append("});")
        return "\n".join(output)

    def print_group(self, group: Group) -> str:
        output = []
        output.append("NCCLCHECK(ncclGroupStart());")
        for command in group.commands:
            output.append(self.print_command(command))
        output.append("NCCLCHECK(ncclGroupEnd());")
        return "\n".join(output)

    def print_command(self, command: Command) -> str:
        match command:
            case PointToPoint() as command:
                return self.print_send_recv(command)
            case Collective() as command:
                return self.print_collective(command)
            case Group() as group:
                return self.print_group(group)
            case _:
                raise TypeError(f"Unknown command type: {type(command)}")

    def print_send_recv(self, command: PointToPoint) -> str:
        return self._generate_output(command, is_point_to_point=True)

    def print_collective(
        self,
        command: Collective,
    ) -> str:
        return self._generate_output(
            command,
            is_point_to_point=False,
            is_rooted=isinstance(command, Rooted),
            is_reductive=isinstance(command, Reductive)
        )

    def _generate_output(
        self,
        command: Command,
        is_point_to_point: bool = False,
        is_rooted: bool = False,
        is_reductive: bool = False
    ) -> str:
        
        device = command.rank
        cuda_set_device_str = ""
        if device != self.current_device:
            cuda_set_device_str = f"CUDACHECK(cudaSetDevice({device}));\n"
            self.current_device = device
        
        buffer_type_str = command.meta.kind.value
        buffer_size = command.meta.size
        rank = command.rank
        
        nccl_comm = f"getComm({command.rank}, {command.channel.communicator})"
        nccl_stream = self._stream_index(device, command.channel.stream)

        
        if is_point_to_point:
            peer_rank = command.peer
            buffer_ptr = f"devBuffers[{device}][{command.buffer}]"
            
            return (
                f"{cuda_set_device_str}"
                f"NCCLCHECK({ncclString(command)}({buffer_ptr}, {buffer_size}, nccl{buffer_type_str.capitalize()}, {peer_rank}, {nccl_comm}, {nccl_stream}));"
            )
        else:
            source_ptr = f"devBuffers[{device}][{command.source_buffer}]"
            destination_ptr = f"devBuffers[{device}][{command.destination_buffer}]"
            
            args = [
                source_ptr,
                destination_ptr,
                str(buffer_size),
                f"nccl{buffer_type_str.capitalize()}",
            ]

            if is_rooted:
                root = command.root 
                args.append(str(root))
            
            if is_reductive:
                op_map = {
                    ReduceOp.Sum: "ncclSum",
                    ReduceOp.Product: "ncclProduct"
                }
                op = op_map.get(command.op)
                if op is None:
                    raise ValueError(f"Unknown reduce operation: {command.op}")
                args.append(op)
            
            args.extend([nccl_comm, nccl_stream])
            
            return (
                f"{cuda_set_device_str}"
                f"NCCLCHECK({ncclString(command)}({', '.join(args)}));"
            )
