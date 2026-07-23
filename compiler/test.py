import unittest
from .parser import program
from .dsl_ast import (
    Program, Thread, AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, AllToAll, Gather, Scatter,
    Channel, Metadata, BufferType, ReduceOp, Send, Recv, PointToPoint, Command, Group
)
from .printer import Printer

class TestParser(unittest.TestCase):
    def test_broadcast(self):
        example = "thread { broadcast<1,2,3>[4,5](int,100,0) }"
        expected = Program(threads=[
            Thread(commands=[
                Broadcast(
                    rank=1,
                    channel=Channel(communicator=2, stream=3),
                    source_buffer=4,
                    destination_buffer=5,
                    meta=Metadata(kind=BufferType.Int, size=100),
                    root=0
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_reduce(self):
        example = "thread { reduce<1,2,3>[4,5](int,100,0,sum) }"
        expected = Program(threads=[
            Thread(commands=[
                Reduce(
                    rank=1,
                    channel=Channel(communicator=2, stream=3),
                    source_buffer=4,
                    destination_buffer=5,
                    meta=Metadata(kind=BufferType.Int, size=100),
                    root=0,
                    op=ReduceOp.Sum
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_all_gather(self):
        example = "thread { allGather<1,2,3>[4,5](int,100) }"
        expected = Program(threads=[
            Thread(commands=[
                AllGather(
                    rank=1,
                    channel=Channel(communicator=2, stream=3),
                    source_buffer=4,
                    destination_buffer=5,
                    meta=Metadata(kind=BufferType.Int, size=100)
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_reduce_scatter(self):
        example = "thread { reduceScatter<1,2,3>[4,5](int,100,product) }"
        expected = Program(threads=[
            Thread(commands=[
                ReduceScatter(
                    rank=1,
                    channel=Channel(communicator=2, stream=3),
                    source_buffer=4,
                    destination_buffer=5,
                    meta=Metadata(kind=BufferType.Int, size=100),
                    op=ReduceOp.Product
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_all_to_all(self):
        example = "thread { allToAll<1,2,3>[4,5](float,200) }"
        expected = Program(threads=[
            Thread(commands=[
                AllToAll(
                    rank=1,
                    channel=Channel(communicator=2, stream=3),
                    source_buffer=4,
                    destination_buffer=5,
                    meta=Metadata(kind=BufferType.Float, size=200)
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_gather(self):
        example = "thread { gather<1,2,3>[4,5](int,100,0) }"
        expected = Program(threads=[
            Thread(commands=[
                Gather(
                    rank=1,
                    channel=Channel(communicator=2, stream=3),
                    source_buffer=4,
                    destination_buffer=5,
                    meta=Metadata(kind=BufferType.Int, size=100),
                    root=0
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_scatter(self):
        example = "thread { scatter<1,2,3>[4,5](int,100,0) }"
        expected = Program(threads=[
            Thread(commands=[
                Scatter(
                    rank=1,
                    channel=Channel(communicator=2, stream=3),
                    source_buffer=4,
                    destination_buffer=5,
                    meta=Metadata(kind=BufferType.Int, size=100),
                    root=0
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_multiple_commands(self):
        example = "thread { allReduce<1,2,3>[4,5](int,100,sum); broadcast<1,2,3>[4,5](int,100,0) }"
        expected = Program(threads=[
            Thread(commands=[
                AllReduce(
                    rank=1,
                    channel=Channel(communicator=2, stream=3),
                    source_buffer=4,
                    destination_buffer=5,
                    meta=Metadata(kind=BufferType.Int, size=100),
                    op=ReduceOp.Sum
                ),
                Broadcast(
                    rank=1,
                    channel=Channel(communicator=2, stream=3),
                    source_buffer=4,
                    destination_buffer=5,
                    meta=Metadata(kind=BufferType.Int, size=100),
                    root=0
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_whitespace_tolerance(self):
        example = "  thread { allReduce < 1 , 2 , 3 > [ 4 , 5 ] ( int , 100 , sum ) ;  broadcast < 1, 2, 3 > [ 4, 5 ] ( int, 100, 0 ) } "
        expected = Program(threads=[
            Thread(commands=[
                AllReduce(
                    rank=1,
                    channel=Channel(communicator=2, stream=3),
                    source_buffer=4,
                    destination_buffer=5,
                    meta=Metadata(kind=BufferType.Int, size=100),
                    op=ReduceOp.Sum
                ),
                Broadcast(
                    rank=1,
                    channel=Channel(communicator=2, stream=3),
                    source_buffer=4,
                    destination_buffer=5,
                    meta=Metadata(kind=BufferType.Int, size=100),
                    root=0
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_send(self):
        example = "thread { send<0,1,2,3>[4](int,100) }"
        expected = Program(threads=[
            Thread(commands=[
                Send(
                    rank=0,
                    peer=1,
                    channel=Channel(communicator=2, stream=3),
                    buffer=4,
                    meta=Metadata(kind=BufferType.Int, size=100)
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_recv(self):
        example = "thread { recv<1,0,2,3>[4](float,200) }"
        expected = Program(threads=[
            Thread(commands=[
                Recv(
                    rank=1,
                    peer=0,
                    channel=Channel(communicator=2, stream=3),
                    buffer=4,
                    meta=Metadata(kind=BufferType.Float, size=200)
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_multiple_send_recv(self):
        example = "thread { send<0,1,2,3>[4](int,100); recv<1,0,2,3>[4](float,200) }"
        expected = Program(threads=[
            Thread(commands=[
                Send(
                    rank=0,
                    peer=1,
                    channel=Channel(communicator=2, stream=3),
                    buffer=4,
                    meta=Metadata(kind=BufferType.Int, size=100)
                ),
                Recv(
                    rank=1,
                    peer=0,
                    channel=Channel(communicator=2, stream=3),
                    buffer=4,
                    meta=Metadata(kind=BufferType.Float, size=200)
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_multiple_threads(self):
        example = "thread { send<0,1,2,3>[4](int,100) } thread { recv<1,0,2,3>[4](int,100) }"
        expected = Program(threads=[
            Thread(commands=[
                Send(
                    rank=0,
                    peer=1,
                    channel=Channel(communicator=2, stream=3),
                    buffer=4,
                    meta=Metadata(kind=BufferType.Int, size=100)
                )
            ]),
            Thread(commands=[
                Recv(
                    rank=1,
                    peer=0,
                    channel=Channel(communicator=2, stream=3),
                    buffer=4,
                    meta=Metadata(kind=BufferType.Int, size=100)
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_multi_digit_identifiers(self):
        example = "thread { send<10,11,12,13>[14](int,100) }"
        expected = Program(threads=[
            Thread(commands=[
                Send(
                    rank=10,
                    peer=11,
                    channel=Channel(communicator=12, stream=13),
                    buffer=14,
                    meta=Metadata(kind=BufferType.Int, size=100)
                )
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_group_syntax(self):
        example = "thread { group { send<0,1,0,0>[0](int,100); recv<1,0,0,1>[0](int,100) } }"
        expected = Program(threads=[
            Thread(commands=[
                Group(commands=[
                    Send(
                        rank=0,
                        peer=1,
                        channel=Channel(communicator=0, stream=0),
                        buffer=0,
                        meta=Metadata(kind=BufferType.Int, size=100)
                    ),
                    Recv(
                        rank=1,
                        peer=0,
                        channel=Channel(communicator=0, stream=1),
                        buffer=0,
                        meta=Metadata(kind=BufferType.Int, size=100)
                    )
                ])
            ])
        ])
        result = program.parse(example)
        self.assertEqual(result, expected)

    def test_nested_groups_and_commands(self):
        example = """
        thread {
            allReduce<0,0,0>[0,1](float, 10, sum);
            group {
                broadcast<0,0,1>[0,0](float, 10, 0);
                send<0,1,0,2>[1](float, 10);
            }
        }
        """
        result = program.parse(example)
        self.assertEqual(len(result.threads), 1)
        self.assertEqual(len(result.threads[0].commands), 2)
        self.assertIsInstance(result.threads[0].commands[1], Group)
        self.assertEqual(len(result.threads[0].commands[1].commands), 2)

class TestPrinter(unittest.TestCase):
    def setUp(self):
        self.printer = Printer(streams_per_gpu=1)

    def test_print_send(self):
        command = Send(
            rank=0,
            peer=1,
            channel=Channel(communicator=0, stream=0),
            buffer=0,
            meta=Metadata(kind=BufferType.Int, size=100)
        )
        program_obj = Program(threads=[Thread(commands=[command])])
        expected_output = (
            "std::barrier sync_point(1);\n"
            "std::vector<std::jthread> threads;\n"
            "threads.emplace_back([&]() {\n"
            "    sync_point.arrive_and_wait();\n"
            "    CUDACHECK(cudaSetDevice(0));\n"
            "    NCCLCHECK(ncclSend(devBuffers[0][0], 100, ncclInt, 1, getComm(0, 0), getStream(0, 0)));\n"
            "});"
        )
        self.assertEqual(self.printer.print_program(program_obj), expected_output)

    def test_print_recv(self):
        command = Recv(
            rank=0,
            peer=1,
            channel=Channel(communicator=0, stream=0),
            buffer=0,
            meta=Metadata(kind=BufferType.Int, size=100)
        )
        program_obj = Program(threads=[Thread(commands=[command])])
        expected_output = (
            "std::barrier sync_point(1);\n"
            "std::vector<std::jthread> threads;\n"
            "threads.emplace_back([&]() {\n"
            "    sync_point.arrive_and_wait();\n"
            "    CUDACHECK(cudaSetDevice(0));\n"
            "    NCCLCHECK(ncclRecv(devBuffers[0][0], 100, ncclInt, 1, getComm(0, 0), getStream(0, 0)));\n"
            "});"
        )
        self.assertEqual(self.printer.print_program(program_obj), expected_output)

    def test_print_all_reduce(self):
        command = AllReduce(
            rank=0,
            channel=Channel(communicator=0, stream=0),
            source_buffer=0,
            destination_buffer=1,
            meta=Metadata(kind=BufferType.Int, size=100),
            op=ReduceOp.Sum
        )
        program_obj = Program(threads=[Thread(commands=[command])])
        expected_output = (
            "std::barrier sync_point(1);\n"
            "std::vector<std::jthread> threads;\n"
            "threads.emplace_back([&]() {\n"
            "    sync_point.arrive_and_wait();\n"
            "    CUDACHECK(cudaSetDevice(0));\n"
            "    NCCLCHECK(ncclAllReduce(devBuffers[0][0], devBuffers[0][1], 100, ncclInt, ncclSum, getComm(0, 0), getStream(0, 0)));\n"
            "});"
        )
        self.assertEqual(self.printer.print_program(program_obj), expected_output)

    def test_print_multiple_commands_with_device_change(self):
        command1 = Send(
            rank=0,
            peer=1,
            channel=Channel(communicator=0, stream=0),
            buffer=0,
            meta=Metadata(kind=BufferType.Int, size=100)
        )
        command2 = Send(
            rank=0,
            peer=1,
            channel=Channel(communicator=1, stream=1),
            buffer=1,
            meta=Metadata(kind=BufferType.Int, size=200)
        )
        program_obj = Program(threads=[Thread(commands=[command1, command2])])
        expected_output = (
            "std::barrier sync_point(1);\n"
            "std::vector<std::jthread> threads;\n"
            "threads.emplace_back([&]() {\n"
            "    sync_point.arrive_and_wait();\n"
            "    CUDACHECK(cudaSetDevice(0));\n"
            "    NCCLCHECK(ncclSend(devBuffers[0][0], 100, ncclInt, 1, getComm(0, 0), getStream(0, 0)));\n"
            "    NCCLCHECK(ncclSend(devBuffers[0][1], 200, ncclInt, 1, getComm(0, 1), getStream(0, 1)));\n"
            "});"
        )
        self.assertEqual(self.printer.print_program(program_obj), expected_output)

    def test_print_multiple_commands_same_device(self):
        command1 = Send(
            rank=0,
            peer=1,
            channel=Channel(communicator=0, stream=0),
            buffer=0,
            meta=Metadata(kind=BufferType.Int, size=100)
        )
        command2 = Send(
            rank=0,
            peer=1,
            channel=Channel(communicator=0, stream=1),
            buffer=1,
            meta=Metadata(kind=BufferType.Int, size=200)
        )
        program_obj = Program(threads=[Thread(commands=[command1, command2])])
        expected_output = (
            "std::barrier sync_point(1);\n"
            "std::vector<std::jthread> threads;\n"
            "threads.emplace_back([&]() {\n"
            "    sync_point.arrive_and_wait();\n"
            "    CUDACHECK(cudaSetDevice(0));\n"
            "    NCCLCHECK(ncclSend(devBuffers[0][0], 100, ncclInt, 1, getComm(0, 0), getStream(0, 0)));\n"
            "    NCCLCHECK(ncclSend(devBuffers[0][1], 200, ncclInt, 1, getComm(0, 0), getStream(0, 1)));\n"
            "});"
        )
        self.assertEqual(self.printer.print_program(program_obj), expected_output)

    def test_print_multiple_threads(self):
        command1 = Send(
            rank=0,
            peer=1,
            channel=Channel(communicator=0, stream=0),
            buffer=0,
            meta=Metadata(kind=BufferType.Int, size=100)
        )
        command2 = Recv(
            rank=1,
            peer=0,
            channel=Channel(communicator=1, stream=1),
            buffer=1,
            meta=Metadata(kind=BufferType.Int, size=100)
        )
        program_obj = Program(threads=[
            Thread(commands=[command1]),
            Thread(commands=[command2])
        ])
        expected_output = (
            "std::barrier sync_point(2);\n"
            "std::vector<std::jthread> threads;\n"
            "threads.emplace_back([&]() {\n"
            "    sync_point.arrive_and_wait();\n"
            "    CUDACHECK(cudaSetDevice(0));\n"
            "    NCCLCHECK(ncclSend(devBuffers[0][0], 100, ncclInt, 1, getComm(0, 0), getStream(0, 0)));\n"
            "});\n"
            "threads.emplace_back([&]() {\n"
            "    sync_point.arrive_and_wait();\n"
            "    CUDACHECK(cudaSetDevice(1));\n"
            "    NCCLCHECK(ncclRecv(devBuffers[1][1], 100, ncclInt, 0, getComm(1, 1), getStream(1, 1)));\n"
            "});"
        )
        self.assertEqual(self.printer.print_program(program_obj), expected_output)

    def test_print_group(self):
        cmd1 = Send(rank=0, peer=1, channel=Channel(0, 0), buffer=0, meta=Metadata(BufferType.Int, 100))
        cmd2 = Send(rank=0, peer=2, channel=Channel(0, 1), buffer=0, meta=Metadata(BufferType.Int, 100))
        group = Group(commands=[cmd1, cmd2])
        program_obj = Program(threads=[Thread(commands=[group])])
        expected_output = (
            "std::barrier sync_point(1);\n"
            "std::vector<std::jthread> threads;\n"
            "threads.emplace_back([&]() {\n"
            "    sync_point.arrive_and_wait();\n"
            "    NCCLCHECK(ncclGroupStart());\n"
            "    CUDACHECK(cudaSetDevice(0));\n"
            "    NCCLCHECK(ncclSend(devBuffers[0][0], 100, ncclInt, 1, getComm(0, 0), getStream(0, 0)));\n"
            "    NCCLCHECK(ncclSend(devBuffers[0][0], 100, ncclInt, 2, getComm(0, 0), getStream(0, 1)));\n"
            "    NCCLCHECK(ncclGroupEnd());\n"
            "});"
        )
        self.assertEqual(self.printer.print_program(program_obj), expected_output)

    def test_print_group_with_device_change(self):
        cmd1 = AllReduce(rank=0, channel=Channel(0,0), source_buffer=0, destination_buffer=1, meta=Metadata(BufferType.Float, 10), op=ReduceOp.Sum)
        cmd2 = Broadcast(rank=0, channel=Channel(0,1), source_buffer=0, destination_buffer=0, meta=Metadata(BufferType.Float, 10), root=0)
        group = Group(commands=[cmd1, cmd2])
        program_obj = Program(threads=[Thread(commands=[group])])
        
        output = self.printer.print_program(program_obj)
        self.assertIn("ncclGroupStart()", output)
        self.assertIn("ncclGroupEnd()", output)
        self.assertIn("cudaSetDevice(0)", output)
        self.assertEqual(output.count("cudaSetDevice(0)"), 1) # Should only set once inside/before group if same device

    def test_print_complex_overlap(self):
        # Thread 0: Rank 0, Comm 0, Stream 0
        # Thread 1: Rank 1, Comm 0, Stream 0
        # Thread 2: Rank 0, Comm 1, Stream 1
        cmd1 = Send(rank=0, peer=1, channel=Channel(0, 0), buffer=0, meta=Metadata(BufferType.Int, 100))
        cmd2 = Recv(rank=1, peer=0, channel=Channel(0, 0), buffer=0, meta=Metadata(BufferType.Int, 100))
        cmd3 = Broadcast(rank=0, channel=Channel(1, 1), source_buffer=0, destination_buffer=1, meta=Metadata(BufferType.Int, 100), root=0)
        
        program_obj = Program(threads=[
            Thread(commands=[cmd1]),
            Thread(commands=[cmd2]),
            Thread(commands=[cmd3])
        ])
        
        expected_output = (
            "std::barrier sync_point(3);\n"
            "std::vector<std::jthread> threads;\n"
            "threads.emplace_back([&]() {\n"
            "    sync_point.arrive_and_wait();\n"
            "    CUDACHECK(cudaSetDevice(0));\n"
            "    NCCLCHECK(ncclSend(devBuffers[0][0], 100, ncclInt, 1, getComm(0, 0), getStream(0, 0)));\n"
            "});\n"
            "threads.emplace_back([&]() {\n"
            "    sync_point.arrive_and_wait();\n"
            "    CUDACHECK(cudaSetDevice(1));\n"
            "    NCCLCHECK(ncclRecv(devBuffers[1][0], 100, ncclInt, 0, getComm(1, 0), getStream(1, 0)));\n"
            "});\n"
            "threads.emplace_back([&]() {\n"
            "    sync_point.arrive_and_wait();\n"
            "    CUDACHECK(cudaSetDevice(0));\n"
            "    NCCLCHECK(ncclBroadcast(devBuffers[0][0], devBuffers[0][1], 100, ncclInt, 0, getComm(0, 1), getStream(0, 1)));\n"
            "});"
        )
        self.assertEqual(self.printer.print_program(program_obj), expected_output)


if __name__ == '__main__':
    unittest.main()
