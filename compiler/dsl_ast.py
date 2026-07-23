from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Generic, TypeVar
from abc import ABC
from parsy import from_enum, regex, seq, string, decimal_digit
from functools import partial

T = TypeVar('T')

class BufferSize(Enum):
    BufferSize = None
    def __str__(self):
        return "size"

class NonEmptyList(Generic[T]):
    def __init__(self, head: T, *tail: T):
        self.items = [head] + list(tail)

class BufferType(Enum):
    Int = "int"
    Float = "float"

@dataclass
class Metadata():
    kind: BufferType
    size: int | BufferSize

@dataclass
class Channel():
    communicator: int
    stream: int

@dataclass(frozen=True)
class Context:
    indent_level: int
    ranks: int
    buffers: int
    current_device: int

class ReduceOp(Enum):
    Sum = "sum"
    Product = "product"

class CollectiveFunction(Enum):
    AllReduce = "allReduce"
    Broadcast = "broadcast"
    Reduce = "reduce"
    AllGather = "allGather"
    ReduceScatter = "reduceScatter"
    AllToAll = "allToAll"
    Gather = "gather"
    Scatter = "scatter"

    @property
    def ncclString(self):
        mapping = {
            CollectiveFunction.AllReduce: "ncclAllReduce",
            CollectiveFunction.Reduce: "ncclReduce",
            CollectiveFunction.AllGather: "ncclAllGather",
            CollectiveFunction.ReduceScatter: "ncclReduceScatter",
            CollectiveFunction.AllToAll: "ncclAllToAll",
            CollectiveFunction.Gather: "ncclGather",
            CollectiveFunction.Scatter: "ncclScatter",
        }
        return mapping[self]

class PointToPointFunction(Enum):
    Send = "send"
    Receive = "recv"

    @property
    def ncclString(self):
        mapping = {
            PointToPointFunction.Send: "ncclSend",
            PointToPointFunction.Receive: "ncclRecv",
        }
        return mapping[self]

    
@dataclass
class Node(ABC):
    pass

@dataclass
class Command(Node):
    pass

@dataclass
class Collective(Command):
    rank: int
    channel: Channel
    source_buffer: int
    destination_buffer: int
    meta: Metadata

@dataclass
class Rooted:
    root: int

@dataclass
class Reductive:
    op: ReduceOp

@dataclass
class AllReduce(Collective, Reductive):
    pass

@dataclass
class Broadcast(Collective, Rooted):
    pass

@dataclass
class Reduce(Collective, Rooted, Reductive):
    pass

@dataclass 
class AllGather(Collective):
    pass

@dataclass
class ReduceScatter(Collective, Reductive):
    pass

@dataclass
class AllToAll(Collective):
    pass

@dataclass
class Gather(Collective, Rooted):
    pass

@dataclass
class Scatter(Collective, Rooted):
    pass

@dataclass
class PointToPoint(Command):
    rank: int
    peer: int
    channel: Channel
    buffer: int
    meta: Metadata

@dataclass
class Send(PointToPoint):
    pass

@dataclass
class Recv(PointToPoint):
    pass

@dataclass
class Group(Node):
    commands: NonEmptyList(Command)

@dataclass
class Thread(Node):
    commands: NonEmptyList(Command | Group)

@dataclass
class Program(Node):
    threads: NonEmptyList(Thread)
