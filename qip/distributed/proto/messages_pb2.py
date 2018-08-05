# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: messages.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='messages.proto',
  package='distributed',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x0emessages.proto\x12\x0b\x64istributed\"3\n\rComplexVector\x12\x10\n\x04real\x18\x01 \x03(\x01\x42\x02\x10\x01\x12\x10\n\x04imag\x18\x02 \x03(\x01\x42\x02\x10\x01\"U\n\rComplexMatrix\x12\x0c\n\x04\x63ols\x18\x01 \x01(\x05\x12\x0c\n\x04rows\x18\x02 \x01(\x05\x12(\n\x04\x64\x61ta\x18\x03 \x01(\x0b\x32\x1a.distributed.ComplexVector\"\x18\n\x07Indices\x12\r\n\x05index\x18\x01 \x03(\x05\"v\n\x05State\x12%\n\x07indices\x18\x01 \x01(\x0b\x32\x14.distributed.Indices\x12,\n\x06vector\x18\x02 \x01(\x0b\x32\x1a.distributed.ComplexVectorH\x00\x12\x0f\n\x05index\x18\x03 \x01(\x03H\x00\x42\x07\n\x05state\";\n\nStateSetup\x12\t\n\x01n\x18\x01 \x01(\x05\x12\"\n\x06states\x18\x02 \x03(\x0b\x32\x12.distributed.State\"I\n\x0cStateHandler\x12\x16\n\x0cstate_handle\x18\x01 \x01(\tH\x00\x12\x17\n\rerror_message\x18\x02 \x01(\tH\x00\x42\x08\n\x06handle\"\x86\x01\n\x0bWorkerSetup\x12\t\n\x01n\x18\x01 \x01(\x05\x12\"\n\x06states\x18\x02 \x03(\x0b\x32\x12.distributed.State\x12\x19\n\x11state_index_start\x18\x03 \x01(\x03\x12\x17\n\x0fstate_index_end\x18\x04 \x01(\x03\x12\x14\n\x0cstate_handle\x18\x05 \x01(\t\"]\n\x08MatrixOp\x12%\n\x07indices\x18\x01 \x01(\x0b\x32\x14.distributed.Indices\x12*\n\x06matrix\x18\x02 \x01(\x0b\x32\x1a.distributed.ComplexMatrix\"3\n\x08KronProd\x12\'\n\x08matrices\x18\x01 \x03(\x0b\x32\x15.distributed.MatrixOp\"0\n\x07Measure\x12%\n\x07indices\x18\x01 \x01(\x0b\x32\x14.distributed.Indices\"\x99\x01\n\x0fWorkerOperation\x12\x0e\n\x06job_id\x18\x01 \x01(\t\x12\x0c\n\x04\x64one\x18\x02 \x01(\x08\x12)\n\x08kronprod\x18\x03 \x01(\x0b\x32\x15.distributed.KronProdH\x00\x12\'\n\x07measure\x18\x04 \x01(\x0b\x32\x14.distributed.MeasureH\x00\x12\x0e\n\x04sync\x18\x05 \x01(\x08H\x00\x42\x04\n\x02op\"\x1f\n\rWorkerConfirm\x12\x0e\n\x06job_id\x18\x01 \x01(\t')
)




_COMPLEXVECTOR = _descriptor.Descriptor(
  name='ComplexVector',
  full_name='distributed.ComplexVector',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='real', full_name='distributed.ComplexVector.real', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='imag', full_name='distributed.ComplexVector.imag', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=31,
  serialized_end=82,
)


_COMPLEXMATRIX = _descriptor.Descriptor(
  name='ComplexMatrix',
  full_name='distributed.ComplexMatrix',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='cols', full_name='distributed.ComplexMatrix.cols', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rows', full_name='distributed.ComplexMatrix.rows', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='distributed.ComplexMatrix.data', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=84,
  serialized_end=169,
)


_INDICES = _descriptor.Descriptor(
  name='Indices',
  full_name='distributed.Indices',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='index', full_name='distributed.Indices.index', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=171,
  serialized_end=195,
)


_STATE = _descriptor.Descriptor(
  name='State',
  full_name='distributed.State',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='indices', full_name='distributed.State.indices', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vector', full_name='distributed.State.vector', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='index', full_name='distributed.State.index', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='state', full_name='distributed.State.state',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=197,
  serialized_end=315,
)


_STATESETUP = _descriptor.Descriptor(
  name='StateSetup',
  full_name='distributed.StateSetup',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='n', full_name='distributed.StateSetup.n', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='states', full_name='distributed.StateSetup.states', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=317,
  serialized_end=376,
)


_STATEHANDLER = _descriptor.Descriptor(
  name='StateHandler',
  full_name='distributed.StateHandler',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='state_handle', full_name='distributed.StateHandler.state_handle', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='error_message', full_name='distributed.StateHandler.error_message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='handle', full_name='distributed.StateHandler.handle',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=378,
  serialized_end=451,
)


_WORKERSETUP = _descriptor.Descriptor(
  name='WorkerSetup',
  full_name='distributed.WorkerSetup',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='n', full_name='distributed.WorkerSetup.n', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='states', full_name='distributed.WorkerSetup.states', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='state_index_start', full_name='distributed.WorkerSetup.state_index_start', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='state_index_end', full_name='distributed.WorkerSetup.state_index_end', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='state_handle', full_name='distributed.WorkerSetup.state_handle', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=454,
  serialized_end=588,
)


_MATRIXOP = _descriptor.Descriptor(
  name='MatrixOp',
  full_name='distributed.MatrixOp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='indices', full_name='distributed.MatrixOp.indices', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='matrix', full_name='distributed.MatrixOp.matrix', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=590,
  serialized_end=683,
)


_KRONPROD = _descriptor.Descriptor(
  name='KronProd',
  full_name='distributed.KronProd',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='matrices', full_name='distributed.KronProd.matrices', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=685,
  serialized_end=736,
)


_MEASURE = _descriptor.Descriptor(
  name='Measure',
  full_name='distributed.Measure',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='indices', full_name='distributed.Measure.indices', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=738,
  serialized_end=786,
)


_WORKEROPERATION = _descriptor.Descriptor(
  name='WorkerOperation',
  full_name='distributed.WorkerOperation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='job_id', full_name='distributed.WorkerOperation.job_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='done', full_name='distributed.WorkerOperation.done', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kronprod', full_name='distributed.WorkerOperation.kronprod', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='measure', full_name='distributed.WorkerOperation.measure', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sync', full_name='distributed.WorkerOperation.sync', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='op', full_name='distributed.WorkerOperation.op',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=789,
  serialized_end=942,
)


_WORKERCONFIRM = _descriptor.Descriptor(
  name='WorkerConfirm',
  full_name='distributed.WorkerConfirm',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='job_id', full_name='distributed.WorkerConfirm.job_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=944,
  serialized_end=975,
)

_COMPLEXMATRIX.fields_by_name['data'].message_type = _COMPLEXVECTOR
_STATE.fields_by_name['indices'].message_type = _INDICES
_STATE.fields_by_name['vector'].message_type = _COMPLEXVECTOR
_STATE.oneofs_by_name['state'].fields.append(
  _STATE.fields_by_name['vector'])
_STATE.fields_by_name['vector'].containing_oneof = _STATE.oneofs_by_name['state']
_STATE.oneofs_by_name['state'].fields.append(
  _STATE.fields_by_name['index'])
_STATE.fields_by_name['index'].containing_oneof = _STATE.oneofs_by_name['state']
_STATESETUP.fields_by_name['states'].message_type = _STATE
_STATEHANDLER.oneofs_by_name['handle'].fields.append(
  _STATEHANDLER.fields_by_name['state_handle'])
_STATEHANDLER.fields_by_name['state_handle'].containing_oneof = _STATEHANDLER.oneofs_by_name['handle']
_STATEHANDLER.oneofs_by_name['handle'].fields.append(
  _STATEHANDLER.fields_by_name['error_message'])
_STATEHANDLER.fields_by_name['error_message'].containing_oneof = _STATEHANDLER.oneofs_by_name['handle']
_WORKERSETUP.fields_by_name['states'].message_type = _STATE
_MATRIXOP.fields_by_name['indices'].message_type = _INDICES
_MATRIXOP.fields_by_name['matrix'].message_type = _COMPLEXMATRIX
_KRONPROD.fields_by_name['matrices'].message_type = _MATRIXOP
_MEASURE.fields_by_name['indices'].message_type = _INDICES
_WORKEROPERATION.fields_by_name['kronprod'].message_type = _KRONPROD
_WORKEROPERATION.fields_by_name['measure'].message_type = _MEASURE
_WORKEROPERATION.oneofs_by_name['op'].fields.append(
  _WORKEROPERATION.fields_by_name['kronprod'])
_WORKEROPERATION.fields_by_name['kronprod'].containing_oneof = _WORKEROPERATION.oneofs_by_name['op']
_WORKEROPERATION.oneofs_by_name['op'].fields.append(
  _WORKEROPERATION.fields_by_name['measure'])
_WORKEROPERATION.fields_by_name['measure'].containing_oneof = _WORKEROPERATION.oneofs_by_name['op']
_WORKEROPERATION.oneofs_by_name['op'].fields.append(
  _WORKEROPERATION.fields_by_name['sync'])
_WORKEROPERATION.fields_by_name['sync'].containing_oneof = _WORKEROPERATION.oneofs_by_name['op']
DESCRIPTOR.message_types_by_name['ComplexVector'] = _COMPLEXVECTOR
DESCRIPTOR.message_types_by_name['ComplexMatrix'] = _COMPLEXMATRIX
DESCRIPTOR.message_types_by_name['Indices'] = _INDICES
DESCRIPTOR.message_types_by_name['State'] = _STATE
DESCRIPTOR.message_types_by_name['StateSetup'] = _STATESETUP
DESCRIPTOR.message_types_by_name['StateHandler'] = _STATEHANDLER
DESCRIPTOR.message_types_by_name['WorkerSetup'] = _WORKERSETUP
DESCRIPTOR.message_types_by_name['MatrixOp'] = _MATRIXOP
DESCRIPTOR.message_types_by_name['KronProd'] = _KRONPROD
DESCRIPTOR.message_types_by_name['Measure'] = _MEASURE
DESCRIPTOR.message_types_by_name['WorkerOperation'] = _WORKEROPERATION
DESCRIPTOR.message_types_by_name['WorkerConfirm'] = _WORKERCONFIRM
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ComplexVector = _reflection.GeneratedProtocolMessageType('ComplexVector', (_message.Message,), dict(
  DESCRIPTOR = _COMPLEXVECTOR,
  __module__ = 'messages_pb2'
  # @@protoc_insertion_point(class_scope:distributed.ComplexVector)
  ))
_sym_db.RegisterMessage(ComplexVector)

ComplexMatrix = _reflection.GeneratedProtocolMessageType('ComplexMatrix', (_message.Message,), dict(
  DESCRIPTOR = _COMPLEXMATRIX,
  __module__ = 'messages_pb2'
  # @@protoc_insertion_point(class_scope:distributed.ComplexMatrix)
  ))
_sym_db.RegisterMessage(ComplexMatrix)

Indices = _reflection.GeneratedProtocolMessageType('Indices', (_message.Message,), dict(
  DESCRIPTOR = _INDICES,
  __module__ = 'messages_pb2'
  # @@protoc_insertion_point(class_scope:distributed.Indices)
  ))
_sym_db.RegisterMessage(Indices)

State = _reflection.GeneratedProtocolMessageType('State', (_message.Message,), dict(
  DESCRIPTOR = _STATE,
  __module__ = 'messages_pb2'
  # @@protoc_insertion_point(class_scope:distributed.State)
  ))
_sym_db.RegisterMessage(State)

StateSetup = _reflection.GeneratedProtocolMessageType('StateSetup', (_message.Message,), dict(
  DESCRIPTOR = _STATESETUP,
  __module__ = 'messages_pb2'
  # @@protoc_insertion_point(class_scope:distributed.StateSetup)
  ))
_sym_db.RegisterMessage(StateSetup)

StateHandler = _reflection.GeneratedProtocolMessageType('StateHandler', (_message.Message,), dict(
  DESCRIPTOR = _STATEHANDLER,
  __module__ = 'messages_pb2'
  # @@protoc_insertion_point(class_scope:distributed.StateHandler)
  ))
_sym_db.RegisterMessage(StateHandler)

WorkerSetup = _reflection.GeneratedProtocolMessageType('WorkerSetup', (_message.Message,), dict(
  DESCRIPTOR = _WORKERSETUP,
  __module__ = 'messages_pb2'
  # @@protoc_insertion_point(class_scope:distributed.WorkerSetup)
  ))
_sym_db.RegisterMessage(WorkerSetup)

MatrixOp = _reflection.GeneratedProtocolMessageType('MatrixOp', (_message.Message,), dict(
  DESCRIPTOR = _MATRIXOP,
  __module__ = 'messages_pb2'
  # @@protoc_insertion_point(class_scope:distributed.MatrixOp)
  ))
_sym_db.RegisterMessage(MatrixOp)

KronProd = _reflection.GeneratedProtocolMessageType('KronProd', (_message.Message,), dict(
  DESCRIPTOR = _KRONPROD,
  __module__ = 'messages_pb2'
  # @@protoc_insertion_point(class_scope:distributed.KronProd)
  ))
_sym_db.RegisterMessage(KronProd)

Measure = _reflection.GeneratedProtocolMessageType('Measure', (_message.Message,), dict(
  DESCRIPTOR = _MEASURE,
  __module__ = 'messages_pb2'
  # @@protoc_insertion_point(class_scope:distributed.Measure)
  ))
_sym_db.RegisterMessage(Measure)

WorkerOperation = _reflection.GeneratedProtocolMessageType('WorkerOperation', (_message.Message,), dict(
  DESCRIPTOR = _WORKEROPERATION,
  __module__ = 'messages_pb2'
  # @@protoc_insertion_point(class_scope:distributed.WorkerOperation)
  ))
_sym_db.RegisterMessage(WorkerOperation)

WorkerConfirm = _reflection.GeneratedProtocolMessageType('WorkerConfirm', (_message.Message,), dict(
  DESCRIPTOR = _WORKERCONFIRM,
  __module__ = 'messages_pb2'
  # @@protoc_insertion_point(class_scope:distributed.WorkerConfirm)
  ))
_sym_db.RegisterMessage(WorkerConfirm)


_COMPLEXVECTOR.fields_by_name['real']._options = None
_COMPLEXVECTOR.fields_by_name['imag']._options = None
# @@protoc_insertion_point(module_scope)
