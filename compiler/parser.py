from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from abc import ABC 
from parsy import from_enum, regex, seq, string, decimal_digit
from functools import partial
from .dsl_ast import *

standard_whitespace = regex(r"\s+")
line_comment = regex(r"//[^\n]*")
inline_comment = regex(r"/\*[\s\S]*?\*/")
whitespace = (standard_whitespace | line_comment | inline_comment).many()

lexeme = lambda p: p << whitespace
digit_int = lexeme(decimal_digit).map(int)
num = lexeme(regex(r"\d+")).map(int)
size = num | lexeme(string("buffer_size")).map(lambda _: BufferSize.BufferSize)

comma = lexeme(string(","))
semicolon = lexeme(string(";"))
langle, rangle = lexeme(string("<")), lexeme(string(">"))
lsquare, rsquare = lexeme(string("[")), lexeme(string("]"))
lparen, rparen = lexeme(string("(")), lexeme(string(")"))
lbrace, rbrace = lexeme(string("{")), lexeme(string("}"))

in_paren = lambda p: lparen >> p << rparen
in_square = lambda p: lsquare >> p << rsquare
in_angle = lambda p: langle >> p << rangle
in_brace = lambda p: lbrace >> p << rbrace

channel = seq(
    communicator = num << comma,
    stream = num
).combine_dict(Channel)

rank_and_channel = in_angle(seq(
    rank = num << comma,
    channel = channel
))

ranks_and_channel = in_angle(seq(
    rank = num << comma,
    peer = num << comma,
    channel = channel
))

buffer = in_square(seq(
    buffer = num
))

buffers = in_square(seq(
    source_buffer = num << comma,
    destination_buffer = num
))

kind = lexeme(from_enum(BufferType))

meta = seq(
    kind = lparen >> kind << comma,
    size = size
).combine_dict(Metadata)

collective_common = seq(rank_and_channel, buffers, meta).map(lambda x: {**x[0], **x[1], 'meta':x[2]})

reduce_op = lexeme(from_enum(ReduceOp))

plain = seq(common=collective_common << rparen)
reductive = seq(common=collective_common << comma, op=reduce_op << rparen)
rooted = seq(common=collective_common << comma, root=num << rparen)
rooted_reductive = seq(
    common=collective_common << comma,
    root=num << comma,
    op=reduce_op << rparen
)

all_reduce = (lexeme(string("allReduce")) >>
     reductive.map(lambda x: AllReduce(**x['common'], op=x['op'])))

broadcast = (lexeme(string("broadcast")) >>
     rooted.map(lambda x: Broadcast(**x['common'], root=x['root'])))

reduce = (lexeme(string("reduce")) >>
    rooted_reductive.map(lambda x: Reduce(**x['common'], root=x['root'], op=x['op'])))

all_gather = (lexeme(string("allGather")) >>
    plain.map(lambda x: AllGather(**x['common'])))

reduce_scatter = (lexeme(string("reduceScatter")) >>
    reductive.map(lambda x: ReduceScatter(**x['common'], op=x['op'])))

all_to_all = (lexeme(string("allToAll")) >>
    plain.map(lambda x: AllToAll(**x['common'])))

gather = (lexeme(string("gather")) >>
    rooted.map(lambda x: Gather(**x['common'], root=x['root'])))

scatter = (lexeme(string("scatter")) >>
    rooted.map(lambda x: Scatter(**x['common'], root=x['root'])))

send_recv_common = seq(ranks_and_channel, buffer, meta << rparen).map(lambda x: {**x[0], **x[1], 'meta':x[2]})

send = lexeme(string("send")) >> send_recv_common.combine_dict(Send)
recv = lexeme(string("recv")) >> send_recv_common.combine_dict(Recv)

sequential = lambda p: p.sep_by(semicolon) << semicolon.optional()

collective = all_reduce | broadcast | reduce | all_gather | reduce_scatter | all_to_all | gather | scatter
point_to_point = send | recv
command = collective | point_to_point

group = lexeme(string("group")) >> in_brace(sequential(command).map(Group))

body = sequential(command | group)

thread = lexeme(string("thread")) >> in_brace(body.map(Thread))

program = whitespace >> (thread.at_least(1) | body.map(Thread).map(lambda t: [t])).map(Program)
