??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02unknown8??
|
Conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 *
shared_nameConv1/kernel
u
 Conv1/kernel/Read/ReadVariableOpReadVariableOpConv1/kernel*&
_output_shapes
:		 *
dtype0
l

Conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Conv1/bias
e
Conv1/bias/Read/ReadVariableOpReadVariableOp
Conv1/bias*
_output_shapes
: *
dtype0
|
Conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameConv2/kernel
u
 Conv2/kernel/Read/ReadVariableOpReadVariableOpConv2/kernel*&
_output_shapes
: @*
dtype0
l

Conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
Conv2/bias
e
Conv2/bias/Read/ReadVariableOpReadVariableOp
Conv2/bias*
_output_shapes
:@*
dtype0
?
Deconv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameDeconv1/kernel
y
"Deconv1/kernel/Read/ReadVariableOpReadVariableOpDeconv1/kernel*&
_output_shapes
: @*
dtype0
p
Deconv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameDeconv1/bias
i
 Deconv1/bias/Read/ReadVariableOpReadVariableOpDeconv1/bias*
_output_shapes
: *
dtype0
?
Deconv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 *
shared_nameDeconv2/kernel
y
"Deconv2/kernel/Read/ReadVariableOpReadVariableOpDeconv2/kernel*&
_output_shapes
:		 *
dtype0
p
Deconv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameDeconv2/bias
i
 Deconv2/bias/Read/ReadVariableOpReadVariableOpDeconv2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api

$	keras_api
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
?
%non_trainable_variables
&layer_regularization_losses
'metrics
(layer_metrics
	variables
trainable_variables
	regularization_losses

)layers
 
XV
VARIABLE_VALUEConv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
Conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
*non_trainable_variables
+layer_regularization_losses
,metrics
-layer_metrics
	variables
trainable_variables
regularization_losses

.layers
XV
VARIABLE_VALUEConv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
Conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
/non_trainable_variables
0layer_regularization_losses
1metrics
2layer_metrics
	variables
trainable_variables
regularization_losses

3layers
ZX
VARIABLE_VALUEDeconv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDeconv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
4non_trainable_variables
5layer_regularization_losses
6metrics
7layer_metrics
	variables
trainable_variables
regularization_losses

8layers
ZX
VARIABLE_VALUEDeconv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDeconv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
9non_trainable_variables
:layer_regularization_losses
;metrics
<layer_metrics
 	variables
!trainable_variables
"regularization_losses

=layers
 
 
 
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_InputLayerPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_InputLayerConv1/kernel
Conv1/biasConv2/kernel
Conv2/biasDeconv1/kernelDeconv1/biasDeconv2/kernelDeconv2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_5691
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename Conv1/kernel/Read/ReadVariableOpConv1/bias/Read/ReadVariableOp Conv2/kernel/Read/ReadVariableOpConv2/bias/Read/ReadVariableOp"Deconv1/kernel/Read/ReadVariableOp Deconv1/bias/Read/ReadVariableOp"Deconv2/kernel/Read/ReadVariableOp Deconv2/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference__traced_save_5936
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv1/kernel
Conv1/biasConv2/kernel
Conv2/biasDeconv1/kernelDeconv1/biasDeconv2/kernelDeconv2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_restore_5970??
?
?
C__inference_fcn_model_layer_call_and_return_conditional_losses_5668

inputlayer$

conv1_5646:		 

conv1_5648: $

conv2_5651: @

conv2_5653:@&
deconv1_5656: @
deconv1_5658: &
deconv2_5661:		 
deconv2_5663:
identity??Conv1/StatefulPartitionedCall?Conv2/StatefulPartitionedCall?Deconv1/StatefulPartitionedCall?Deconv2/StatefulPartitionedCall?
Conv1/StatefulPartitionedCallStatefulPartitionedCall
inputlayer
conv1_5646
conv1_5648*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Conv1_layer_call_and_return_conditional_losses_54562
Conv1/StatefulPartitionedCall?
Conv2/StatefulPartitionedCallStatefulPartitionedCall&Conv1/StatefulPartitionedCall:output:0
conv2_5651
conv2_5653*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Conv2_layer_call_and_return_conditional_losses_54732
Conv2/StatefulPartitionedCall?
Deconv1/StatefulPartitionedCallStatefulPartitionedCall&Conv2/StatefulPartitionedCall:output:0deconv1_5656deconv1_5658*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Deconv1_layer_call_and_return_conditional_losses_53842!
Deconv1/StatefulPartitionedCall?
Deconv2/StatefulPartitionedCallStatefulPartitionedCall(Deconv1/StatefulPartitionedCall:output:0deconv2_5661deconv2_5663*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Deconv2_layer_call_and_return_conditional_losses_54282!
Deconv2/StatefulPartitionedCall?
tf.math.tanh_1/TanhTanh(Deconv2/StatefulPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
tf.math.tanh_1/Tanh?
IdentityIdentitytf.math.tanh_1/Tanh:y:0^Conv1/StatefulPartitionedCall^Conv2/StatefulPartitionedCall ^Deconv1/StatefulPartitionedCall ^Deconv2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2>
Conv1/StatefulPartitionedCallConv1/StatefulPartitionedCall2>
Conv2/StatefulPartitionedCallConv2/StatefulPartitionedCall2B
Deconv1/StatefulPartitionedCallDeconv1/StatefulPartitionedCall2B
Deconv2/StatefulPartitionedCallDeconv2/StatefulPartitionedCall:] Y
1
_output_shapes
:???????????
$
_user_specified_name
InputLayer
?
?
?__inference_Conv1_layer_call_and_return_conditional_losses_5456

inputs8
conv2d_readvariableop_resource:		 -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_Deconv1_layer_call_fn_5394

inputs!
unknown: @
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Deconv1_layer_call_and_return_conditional_losses_53842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?%
?
A__inference_Deconv1_layer_call_and_return_conditional_losses_5384

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
?__inference_Conv2_layer_call_and_return_conditional_losses_5473

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
__inference__traced_save_5936
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop-
)savev2_deconv1_kernel_read_readvariableop+
'savev2_deconv1_bias_read_readvariableop-
)savev2_deconv2_kernel_read_readvariableop+
'savev2_deconv2_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop)savev2_deconv1_kernel_read_readvariableop'savev2_deconv1_bias_read_readvariableop)savev2_deconv2_kernel_read_readvariableop'savev2_deconv2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*w
_input_shapesf
d: :		 : : @:@: @: :		 :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:		 : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
:		 : 

_output_shapes
::	

_output_shapes
: 
?&
?
 __inference__traced_restore_5970
file_prefix7
assignvariableop_conv1_kernel:		 +
assignvariableop_1_conv1_bias: 9
assignvariableop_2_conv2_kernel: @+
assignvariableop_3_conv2_bias:@;
!assignvariableop_4_deconv1_kernel: @-
assignvariableop_5_deconv1_bias: ;
!assignvariableop_6_deconv2_kernel:		 -
assignvariableop_7_deconv2_bias:

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_deconv1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_deconv1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_deconv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_deconv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?$
?
A__inference_Deconv2_layer_call_and_return_conditional_losses_5428

inputsB
(conv2d_transpose_readvariableop_resource:		 -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		 *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?Y
?
__inference__wrapped_model_5349

inputlayerH
.fcn_model_conv1_conv2d_readvariableop_resource:		 =
/fcn_model_conv1_biasadd_readvariableop_resource: H
.fcn_model_conv2_conv2d_readvariableop_resource: @=
/fcn_model_conv2_biasadd_readvariableop_resource:@T
:fcn_model_deconv1_conv2d_transpose_readvariableop_resource: @?
1fcn_model_deconv1_biasadd_readvariableop_resource: T
:fcn_model_deconv2_conv2d_transpose_readvariableop_resource:		 ?
1fcn_model_deconv2_biasadd_readvariableop_resource:
identity??&fcn_model/Conv1/BiasAdd/ReadVariableOp?%fcn_model/Conv1/Conv2D/ReadVariableOp?&fcn_model/Conv2/BiasAdd/ReadVariableOp?%fcn_model/Conv2/Conv2D/ReadVariableOp?(fcn_model/Deconv1/BiasAdd/ReadVariableOp?1fcn_model/Deconv1/conv2d_transpose/ReadVariableOp?(fcn_model/Deconv2/BiasAdd/ReadVariableOp?1fcn_model/Deconv2/conv2d_transpose/ReadVariableOp?
%fcn_model/Conv1/Conv2D/ReadVariableOpReadVariableOp.fcn_model_conv1_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02'
%fcn_model/Conv1/Conv2D/ReadVariableOp?
fcn_model/Conv1/Conv2DConv2D
inputlayer-fcn_model/Conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
fcn_model/Conv1/Conv2D?
&fcn_model/Conv1/BiasAdd/ReadVariableOpReadVariableOp/fcn_model_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&fcn_model/Conv1/BiasAdd/ReadVariableOp?
fcn_model/Conv1/BiasAddBiasAddfcn_model/Conv1/Conv2D:output:0.fcn_model/Conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
fcn_model/Conv1/BiasAdd?
fcn_model/Conv1/ReluRelu fcn_model/Conv1/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
fcn_model/Conv1/Relu?
%fcn_model/Conv2/Conv2D/ReadVariableOpReadVariableOp.fcn_model_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02'
%fcn_model/Conv2/Conv2D/ReadVariableOp?
fcn_model/Conv2/Conv2DConv2D"fcn_model/Conv1/Relu:activations:0-fcn_model/Conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
fcn_model/Conv2/Conv2D?
&fcn_model/Conv2/BiasAdd/ReadVariableOpReadVariableOp/fcn_model_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&fcn_model/Conv2/BiasAdd/ReadVariableOp?
fcn_model/Conv2/BiasAddBiasAddfcn_model/Conv2/Conv2D:output:0.fcn_model/Conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
fcn_model/Conv2/BiasAdd?
fcn_model/Conv2/ReluRelu fcn_model/Conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
fcn_model/Conv2/Relu?
fcn_model/Deconv1/ShapeShape"fcn_model/Conv2/Relu:activations:0*
T0*
_output_shapes
:2
fcn_model/Deconv1/Shape?
%fcn_model/Deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%fcn_model/Deconv1/strided_slice/stack?
'fcn_model/Deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'fcn_model/Deconv1/strided_slice/stack_1?
'fcn_model/Deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'fcn_model/Deconv1/strided_slice/stack_2?
fcn_model/Deconv1/strided_sliceStridedSlice fcn_model/Deconv1/Shape:output:0.fcn_model/Deconv1/strided_slice/stack:output:00fcn_model/Deconv1/strided_slice/stack_1:output:00fcn_model/Deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
fcn_model/Deconv1/strided_slicey
fcn_model/Deconv1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
fcn_model/Deconv1/stack/1y
fcn_model/Deconv1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
fcn_model/Deconv1/stack/2x
fcn_model/Deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
fcn_model/Deconv1/stack/3?
fcn_model/Deconv1/stackPack(fcn_model/Deconv1/strided_slice:output:0"fcn_model/Deconv1/stack/1:output:0"fcn_model/Deconv1/stack/2:output:0"fcn_model/Deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2
fcn_model/Deconv1/stack?
'fcn_model/Deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'fcn_model/Deconv1/strided_slice_1/stack?
)fcn_model/Deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)fcn_model/Deconv1/strided_slice_1/stack_1?
)fcn_model/Deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)fcn_model/Deconv1/strided_slice_1/stack_2?
!fcn_model/Deconv1/strided_slice_1StridedSlice fcn_model/Deconv1/stack:output:00fcn_model/Deconv1/strided_slice_1/stack:output:02fcn_model/Deconv1/strided_slice_1/stack_1:output:02fcn_model/Deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!fcn_model/Deconv1/strided_slice_1?
1fcn_model/Deconv1/conv2d_transpose/ReadVariableOpReadVariableOp:fcn_model_deconv1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype023
1fcn_model/Deconv1/conv2d_transpose/ReadVariableOp?
"fcn_model/Deconv1/conv2d_transposeConv2DBackpropInput fcn_model/Deconv1/stack:output:09fcn_model/Deconv1/conv2d_transpose/ReadVariableOp:value:0"fcn_model/Conv2/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2$
"fcn_model/Deconv1/conv2d_transpose?
(fcn_model/Deconv1/BiasAdd/ReadVariableOpReadVariableOp1fcn_model_deconv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(fcn_model/Deconv1/BiasAdd/ReadVariableOp?
fcn_model/Deconv1/BiasAddBiasAdd+fcn_model/Deconv1/conv2d_transpose:output:00fcn_model/Deconv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
fcn_model/Deconv1/BiasAdd?
fcn_model/Deconv1/ReluRelu"fcn_model/Deconv1/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
fcn_model/Deconv1/Relu?
fcn_model/Deconv2/ShapeShape$fcn_model/Deconv1/Relu:activations:0*
T0*
_output_shapes
:2
fcn_model/Deconv2/Shape?
%fcn_model/Deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%fcn_model/Deconv2/strided_slice/stack?
'fcn_model/Deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'fcn_model/Deconv2/strided_slice/stack_1?
'fcn_model/Deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'fcn_model/Deconv2/strided_slice/stack_2?
fcn_model/Deconv2/strided_sliceStridedSlice fcn_model/Deconv2/Shape:output:0.fcn_model/Deconv2/strided_slice/stack:output:00fcn_model/Deconv2/strided_slice/stack_1:output:00fcn_model/Deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
fcn_model/Deconv2/strided_slicey
fcn_model/Deconv2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
fcn_model/Deconv2/stack/1y
fcn_model/Deconv2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
fcn_model/Deconv2/stack/2x
fcn_model/Deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
fcn_model/Deconv2/stack/3?
fcn_model/Deconv2/stackPack(fcn_model/Deconv2/strided_slice:output:0"fcn_model/Deconv2/stack/1:output:0"fcn_model/Deconv2/stack/2:output:0"fcn_model/Deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2
fcn_model/Deconv2/stack?
'fcn_model/Deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'fcn_model/Deconv2/strided_slice_1/stack?
)fcn_model/Deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)fcn_model/Deconv2/strided_slice_1/stack_1?
)fcn_model/Deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)fcn_model/Deconv2/strided_slice_1/stack_2?
!fcn_model/Deconv2/strided_slice_1StridedSlice fcn_model/Deconv2/stack:output:00fcn_model/Deconv2/strided_slice_1/stack:output:02fcn_model/Deconv2/strided_slice_1/stack_1:output:02fcn_model/Deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!fcn_model/Deconv2/strided_slice_1?
1fcn_model/Deconv2/conv2d_transpose/ReadVariableOpReadVariableOp:fcn_model_deconv2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		 *
dtype023
1fcn_model/Deconv2/conv2d_transpose/ReadVariableOp?
"fcn_model/Deconv2/conv2d_transposeConv2DBackpropInput fcn_model/Deconv2/stack:output:09fcn_model/Deconv2/conv2d_transpose/ReadVariableOp:value:0$fcn_model/Deconv1/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2$
"fcn_model/Deconv2/conv2d_transpose?
(fcn_model/Deconv2/BiasAdd/ReadVariableOpReadVariableOp1fcn_model_deconv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(fcn_model/Deconv2/BiasAdd/ReadVariableOp?
fcn_model/Deconv2/BiasAddBiasAdd+fcn_model/Deconv2/conv2d_transpose:output:00fcn_model/Deconv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
fcn_model/Deconv2/BiasAdd?
fcn_model/tf.math.tanh_1/TanhTanh"fcn_model/Deconv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
fcn_model/tf.math.tanh_1/Tanh?
IdentityIdentity!fcn_model/tf.math.tanh_1/Tanh:y:0'^fcn_model/Conv1/BiasAdd/ReadVariableOp&^fcn_model/Conv1/Conv2D/ReadVariableOp'^fcn_model/Conv2/BiasAdd/ReadVariableOp&^fcn_model/Conv2/Conv2D/ReadVariableOp)^fcn_model/Deconv1/BiasAdd/ReadVariableOp2^fcn_model/Deconv1/conv2d_transpose/ReadVariableOp)^fcn_model/Deconv2/BiasAdd/ReadVariableOp2^fcn_model/Deconv2/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2P
&fcn_model/Conv1/BiasAdd/ReadVariableOp&fcn_model/Conv1/BiasAdd/ReadVariableOp2N
%fcn_model/Conv1/Conv2D/ReadVariableOp%fcn_model/Conv1/Conv2D/ReadVariableOp2P
&fcn_model/Conv2/BiasAdd/ReadVariableOp&fcn_model/Conv2/BiasAdd/ReadVariableOp2N
%fcn_model/Conv2/Conv2D/ReadVariableOp%fcn_model/Conv2/Conv2D/ReadVariableOp2T
(fcn_model/Deconv1/BiasAdd/ReadVariableOp(fcn_model/Deconv1/BiasAdd/ReadVariableOp2f
1fcn_model/Deconv1/conv2d_transpose/ReadVariableOp1fcn_model/Deconv1/conv2d_transpose/ReadVariableOp2T
(fcn_model/Deconv2/BiasAdd/ReadVariableOp(fcn_model/Deconv2/BiasAdd/ReadVariableOp2f
1fcn_model/Deconv2/conv2d_transpose/ReadVariableOp1fcn_model/Deconv2/conv2d_transpose/ReadVariableOp:] Y
1
_output_shapes
:???????????
$
_user_specified_name
InputLayer
?

?
(__inference_fcn_model_layer_call_fn_5618

inputlayer!
unknown:		 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5:		 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputlayerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_fcn_model_layer_call_and_return_conditional_losses_55782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
1
_output_shapes
:???????????
$
_user_specified_name
InputLayer
?
?
C__inference_fcn_model_layer_call_and_return_conditional_losses_5643

inputlayer$

conv1_5621:		 

conv1_5623: $

conv2_5626: @

conv2_5628:@&
deconv1_5631: @
deconv1_5633: &
deconv2_5636:		 
deconv2_5638:
identity??Conv1/StatefulPartitionedCall?Conv2/StatefulPartitionedCall?Deconv1/StatefulPartitionedCall?Deconv2/StatefulPartitionedCall?
Conv1/StatefulPartitionedCallStatefulPartitionedCall
inputlayer
conv1_5621
conv1_5623*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Conv1_layer_call_and_return_conditional_losses_54562
Conv1/StatefulPartitionedCall?
Conv2/StatefulPartitionedCallStatefulPartitionedCall&Conv1/StatefulPartitionedCall:output:0
conv2_5626
conv2_5628*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Conv2_layer_call_and_return_conditional_losses_54732
Conv2/StatefulPartitionedCall?
Deconv1/StatefulPartitionedCallStatefulPartitionedCall&Conv2/StatefulPartitionedCall:output:0deconv1_5631deconv1_5633*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Deconv1_layer_call_and_return_conditional_losses_53842!
Deconv1/StatefulPartitionedCall?
Deconv2/StatefulPartitionedCallStatefulPartitionedCall(Deconv1/StatefulPartitionedCall:output:0deconv2_5636deconv2_5638*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Deconv2_layer_call_and_return_conditional_losses_54282!
Deconv2/StatefulPartitionedCall?
tf.math.tanh_1/TanhTanh(Deconv2/StatefulPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
tf.math.tanh_1/Tanh?
IdentityIdentitytf.math.tanh_1/Tanh:y:0^Conv1/StatefulPartitionedCall^Conv2/StatefulPartitionedCall ^Deconv1/StatefulPartitionedCall ^Deconv2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2>
Conv1/StatefulPartitionedCallConv1/StatefulPartitionedCall2>
Conv2/StatefulPartitionedCallConv2/StatefulPartitionedCall2B
Deconv1/StatefulPartitionedCallDeconv1/StatefulPartitionedCall2B
Deconv2/StatefulPartitionedCallDeconv2/StatefulPartitionedCall:] Y
1
_output_shapes
:???????????
$
_user_specified_name
InputLayer
?
?
C__inference_fcn_model_layer_call_and_return_conditional_losses_5491

inputs$

conv1_5457:		 

conv1_5459: $

conv2_5474: @

conv2_5476:@&
deconv1_5479: @
deconv1_5481: &
deconv2_5484:		 
deconv2_5486:
identity??Conv1/StatefulPartitionedCall?Conv2/StatefulPartitionedCall?Deconv1/StatefulPartitionedCall?Deconv2/StatefulPartitionedCall?
Conv1/StatefulPartitionedCallStatefulPartitionedCallinputs
conv1_5457
conv1_5459*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Conv1_layer_call_and_return_conditional_losses_54562
Conv1/StatefulPartitionedCall?
Conv2/StatefulPartitionedCallStatefulPartitionedCall&Conv1/StatefulPartitionedCall:output:0
conv2_5474
conv2_5476*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Conv2_layer_call_and_return_conditional_losses_54732
Conv2/StatefulPartitionedCall?
Deconv1/StatefulPartitionedCallStatefulPartitionedCall&Conv2/StatefulPartitionedCall:output:0deconv1_5479deconv1_5481*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Deconv1_layer_call_and_return_conditional_losses_53842!
Deconv1/StatefulPartitionedCall?
Deconv2/StatefulPartitionedCallStatefulPartitionedCall(Deconv1/StatefulPartitionedCall:output:0deconv2_5484deconv2_5486*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Deconv2_layer_call_and_return_conditional_losses_54282!
Deconv2/StatefulPartitionedCall?
tf.math.tanh_1/TanhTanh(Deconv2/StatefulPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
tf.math.tanh_1/Tanh?
IdentityIdentitytf.math.tanh_1/Tanh:y:0^Conv1/StatefulPartitionedCall^Conv2/StatefulPartitionedCall ^Deconv1/StatefulPartitionedCall ^Deconv2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2>
Conv1/StatefulPartitionedCallConv1/StatefulPartitionedCall2>
Conv2/StatefulPartitionedCallConv2/StatefulPartitionedCall2B
Deconv1/StatefulPartitionedCallDeconv1/StatefulPartitionedCall2B
Deconv2/StatefulPartitionedCallDeconv2/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?K
?
C__inference_fcn_model_layer_call_and_return_conditional_losses_5749

inputs>
$conv1_conv2d_readvariableop_resource:		 3
%conv1_biasadd_readvariableop_resource: >
$conv2_conv2d_readvariableop_resource: @3
%conv2_biasadd_readvariableop_resource:@J
0deconv1_conv2d_transpose_readvariableop_resource: @5
'deconv1_biasadd_readvariableop_resource: J
0deconv2_conv2d_transpose_readvariableop_resource:		 5
'deconv2_biasadd_readvariableop_resource:
identity??Conv1/BiasAdd/ReadVariableOp?Conv1/Conv2D/ReadVariableOp?Conv2/BiasAdd/ReadVariableOp?Conv2/Conv2D/ReadVariableOp?Deconv1/BiasAdd/ReadVariableOp?'Deconv1/conv2d_transpose/ReadVariableOp?Deconv2/BiasAdd/ReadVariableOp?'Deconv2/conv2d_transpose/ReadVariableOp?
Conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02
Conv1/Conv2D/ReadVariableOp?
Conv1/Conv2DConv2Dinputs#Conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv1/Conv2D?
Conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
Conv1/BiasAdd/ReadVariableOp?
Conv1/BiasAddBiasAddConv1/Conv2D:output:0$Conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
Conv1/BiasAddt

Conv1/ReluReluConv1/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2

Conv1/Relu?
Conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2/Conv2D/ReadVariableOp?
Conv2/Conv2DConv2DConv1/Relu:activations:0#Conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2/Conv2D?
Conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
Conv2/BiasAdd/ReadVariableOp?
Conv2/BiasAddBiasAddConv2/Conv2D:output:0$Conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
Conv2/BiasAddt

Conv2/ReluReluConv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2

Conv2/Reluf
Deconv1/ShapeShapeConv2/Relu:activations:0*
T0*
_output_shapes
:2
Deconv1/Shape?
Deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Deconv1/strided_slice/stack?
Deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Deconv1/strided_slice/stack_1?
Deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Deconv1/strided_slice/stack_2?
Deconv1/strided_sliceStridedSliceDeconv1/Shape:output:0$Deconv1/strided_slice/stack:output:0&Deconv1/strided_slice/stack_1:output:0&Deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Deconv1/strided_slicee
Deconv1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Deconv1/stack/1e
Deconv1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Deconv1/stack/2d
Deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Deconv1/stack/3?
Deconv1/stackPackDeconv1/strided_slice:output:0Deconv1/stack/1:output:0Deconv1/stack/2:output:0Deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2
Deconv1/stack?
Deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Deconv1/strided_slice_1/stack?
Deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
Deconv1/strided_slice_1/stack_1?
Deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
Deconv1/strided_slice_1/stack_2?
Deconv1/strided_slice_1StridedSliceDeconv1/stack:output:0&Deconv1/strided_slice_1/stack:output:0(Deconv1/strided_slice_1/stack_1:output:0(Deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Deconv1/strided_slice_1?
'Deconv1/conv2d_transpose/ReadVariableOpReadVariableOp0deconv1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'Deconv1/conv2d_transpose/ReadVariableOp?
Deconv1/conv2d_transposeConv2DBackpropInputDeconv1/stack:output:0/Deconv1/conv2d_transpose/ReadVariableOp:value:0Conv2/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Deconv1/conv2d_transpose?
Deconv1/BiasAdd/ReadVariableOpReadVariableOp'deconv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
Deconv1/BiasAdd/ReadVariableOp?
Deconv1/BiasAddBiasAdd!Deconv1/conv2d_transpose:output:0&Deconv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
Deconv1/BiasAddz
Deconv1/ReluReluDeconv1/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Deconv1/Reluh
Deconv2/ShapeShapeDeconv1/Relu:activations:0*
T0*
_output_shapes
:2
Deconv2/Shape?
Deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Deconv2/strided_slice/stack?
Deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Deconv2/strided_slice/stack_1?
Deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Deconv2/strided_slice/stack_2?
Deconv2/strided_sliceStridedSliceDeconv2/Shape:output:0$Deconv2/strided_slice/stack:output:0&Deconv2/strided_slice/stack_1:output:0&Deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Deconv2/strided_slicee
Deconv2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Deconv2/stack/1e
Deconv2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Deconv2/stack/2d
Deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Deconv2/stack/3?
Deconv2/stackPackDeconv2/strided_slice:output:0Deconv2/stack/1:output:0Deconv2/stack/2:output:0Deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2
Deconv2/stack?
Deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Deconv2/strided_slice_1/stack?
Deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
Deconv2/strided_slice_1/stack_1?
Deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
Deconv2/strided_slice_1/stack_2?
Deconv2/strided_slice_1StridedSliceDeconv2/stack:output:0&Deconv2/strided_slice_1/stack:output:0(Deconv2/strided_slice_1/stack_1:output:0(Deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Deconv2/strided_slice_1?
'Deconv2/conv2d_transpose/ReadVariableOpReadVariableOp0deconv2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		 *
dtype02)
'Deconv2/conv2d_transpose/ReadVariableOp?
Deconv2/conv2d_transposeConv2DBackpropInputDeconv2/stack:output:0/Deconv2/conv2d_transpose/ReadVariableOp:value:0Deconv1/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Deconv2/conv2d_transpose?
Deconv2/BiasAdd/ReadVariableOpReadVariableOp'deconv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
Deconv2/BiasAdd/ReadVariableOp?
Deconv2/BiasAddBiasAdd!Deconv2/conv2d_transpose:output:0&Deconv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
Deconv2/BiasAdd?
tf.math.tanh_1/TanhTanhDeconv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
tf.math.tanh_1/Tanh?
IdentityIdentitytf.math.tanh_1/Tanh:y:0^Conv1/BiasAdd/ReadVariableOp^Conv1/Conv2D/ReadVariableOp^Conv2/BiasAdd/ReadVariableOp^Conv2/Conv2D/ReadVariableOp^Deconv1/BiasAdd/ReadVariableOp(^Deconv1/conv2d_transpose/ReadVariableOp^Deconv2/BiasAdd/ReadVariableOp(^Deconv2/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2<
Conv1/BiasAdd/ReadVariableOpConv1/BiasAdd/ReadVariableOp2:
Conv1/Conv2D/ReadVariableOpConv1/Conv2D/ReadVariableOp2<
Conv2/BiasAdd/ReadVariableOpConv2/BiasAdd/ReadVariableOp2:
Conv2/Conv2D/ReadVariableOpConv2/Conv2D/ReadVariableOp2@
Deconv1/BiasAdd/ReadVariableOpDeconv1/BiasAdd/ReadVariableOp2R
'Deconv1/conv2d_transpose/ReadVariableOp'Deconv1/conv2d_transpose/ReadVariableOp2@
Deconv2/BiasAdd/ReadVariableOpDeconv2/BiasAdd/ReadVariableOp2R
'Deconv2/conv2d_transpose/ReadVariableOp'Deconv2/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
(__inference_fcn_model_layer_call_fn_5849

inputs!
unknown:		 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5:		 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_fcn_model_layer_call_and_return_conditional_losses_55782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
(__inference_fcn_model_layer_call_fn_5510

inputlayer!
unknown:		 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5:		 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputlayerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_fcn_model_layer_call_and_return_conditional_losses_54912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
1
_output_shapes
:???????????
$
_user_specified_name
InputLayer
?
?
?__inference_Conv1_layer_call_and_return_conditional_losses_5869

inputs8
conv2d_readvariableop_resource:		 -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
"__inference_signature_wrapper_5691

inputlayer!
unknown:		 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5:		 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputlayerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_53492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
1
_output_shapes
:???????????
$
_user_specified_name
InputLayer
?
?
C__inference_fcn_model_layer_call_and_return_conditional_losses_5578

inputs$

conv1_5556:		 

conv1_5558: $

conv2_5561: @

conv2_5563:@&
deconv1_5566: @
deconv1_5568: &
deconv2_5571:		 
deconv2_5573:
identity??Conv1/StatefulPartitionedCall?Conv2/StatefulPartitionedCall?Deconv1/StatefulPartitionedCall?Deconv2/StatefulPartitionedCall?
Conv1/StatefulPartitionedCallStatefulPartitionedCallinputs
conv1_5556
conv1_5558*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Conv1_layer_call_and_return_conditional_losses_54562
Conv1/StatefulPartitionedCall?
Conv2/StatefulPartitionedCallStatefulPartitionedCall&Conv1/StatefulPartitionedCall:output:0
conv2_5561
conv2_5563*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Conv2_layer_call_and_return_conditional_losses_54732
Conv2/StatefulPartitionedCall?
Deconv1/StatefulPartitionedCallStatefulPartitionedCall&Conv2/StatefulPartitionedCall:output:0deconv1_5566deconv1_5568*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Deconv1_layer_call_and_return_conditional_losses_53842!
Deconv1/StatefulPartitionedCall?
Deconv2/StatefulPartitionedCallStatefulPartitionedCall(Deconv1/StatefulPartitionedCall:output:0deconv2_5571deconv2_5573*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Deconv2_layer_call_and_return_conditional_losses_54282!
Deconv2/StatefulPartitionedCall?
tf.math.tanh_1/TanhTanh(Deconv2/StatefulPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
tf.math.tanh_1/Tanh?
IdentityIdentitytf.math.tanh_1/Tanh:y:0^Conv1/StatefulPartitionedCall^Conv2/StatefulPartitionedCall ^Deconv1/StatefulPartitionedCall ^Deconv2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2>
Conv1/StatefulPartitionedCallConv1/StatefulPartitionedCall2>
Conv2/StatefulPartitionedCallConv2/StatefulPartitionedCall2B
Deconv1/StatefulPartitionedCallDeconv1/StatefulPartitionedCall2B
Deconv2/StatefulPartitionedCallDeconv2/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?K
?
C__inference_fcn_model_layer_call_and_return_conditional_losses_5807

inputs>
$conv1_conv2d_readvariableop_resource:		 3
%conv1_biasadd_readvariableop_resource: >
$conv2_conv2d_readvariableop_resource: @3
%conv2_biasadd_readvariableop_resource:@J
0deconv1_conv2d_transpose_readvariableop_resource: @5
'deconv1_biasadd_readvariableop_resource: J
0deconv2_conv2d_transpose_readvariableop_resource:		 5
'deconv2_biasadd_readvariableop_resource:
identity??Conv1/BiasAdd/ReadVariableOp?Conv1/Conv2D/ReadVariableOp?Conv2/BiasAdd/ReadVariableOp?Conv2/Conv2D/ReadVariableOp?Deconv1/BiasAdd/ReadVariableOp?'Deconv1/conv2d_transpose/ReadVariableOp?Deconv2/BiasAdd/ReadVariableOp?'Deconv2/conv2d_transpose/ReadVariableOp?
Conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:		 *
dtype02
Conv1/Conv2D/ReadVariableOp?
Conv1/Conv2DConv2Dinputs#Conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv1/Conv2D?
Conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
Conv1/BiasAdd/ReadVariableOp?
Conv1/BiasAddBiasAddConv1/Conv2D:output:0$Conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
Conv1/BiasAddt

Conv1/ReluReluConv1/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2

Conv1/Relu?
Conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2/Conv2D/ReadVariableOp?
Conv2/Conv2DConv2DConv1/Relu:activations:0#Conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2/Conv2D?
Conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
Conv2/BiasAdd/ReadVariableOp?
Conv2/BiasAddBiasAddConv2/Conv2D:output:0$Conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
Conv2/BiasAddt

Conv2/ReluReluConv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2

Conv2/Reluf
Deconv1/ShapeShapeConv2/Relu:activations:0*
T0*
_output_shapes
:2
Deconv1/Shape?
Deconv1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Deconv1/strided_slice/stack?
Deconv1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Deconv1/strided_slice/stack_1?
Deconv1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Deconv1/strided_slice/stack_2?
Deconv1/strided_sliceStridedSliceDeconv1/Shape:output:0$Deconv1/strided_slice/stack:output:0&Deconv1/strided_slice/stack_1:output:0&Deconv1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Deconv1/strided_slicee
Deconv1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Deconv1/stack/1e
Deconv1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Deconv1/stack/2d
Deconv1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Deconv1/stack/3?
Deconv1/stackPackDeconv1/strided_slice:output:0Deconv1/stack/1:output:0Deconv1/stack/2:output:0Deconv1/stack/3:output:0*
N*
T0*
_output_shapes
:2
Deconv1/stack?
Deconv1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Deconv1/strided_slice_1/stack?
Deconv1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
Deconv1/strided_slice_1/stack_1?
Deconv1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
Deconv1/strided_slice_1/stack_2?
Deconv1/strided_slice_1StridedSliceDeconv1/stack:output:0&Deconv1/strided_slice_1/stack:output:0(Deconv1/strided_slice_1/stack_1:output:0(Deconv1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Deconv1/strided_slice_1?
'Deconv1/conv2d_transpose/ReadVariableOpReadVariableOp0deconv1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'Deconv1/conv2d_transpose/ReadVariableOp?
Deconv1/conv2d_transposeConv2DBackpropInputDeconv1/stack:output:0/Deconv1/conv2d_transpose/ReadVariableOp:value:0Conv2/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Deconv1/conv2d_transpose?
Deconv1/BiasAdd/ReadVariableOpReadVariableOp'deconv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
Deconv1/BiasAdd/ReadVariableOp?
Deconv1/BiasAddBiasAdd!Deconv1/conv2d_transpose:output:0&Deconv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
Deconv1/BiasAddz
Deconv1/ReluReluDeconv1/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Deconv1/Reluh
Deconv2/ShapeShapeDeconv1/Relu:activations:0*
T0*
_output_shapes
:2
Deconv2/Shape?
Deconv2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Deconv2/strided_slice/stack?
Deconv2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Deconv2/strided_slice/stack_1?
Deconv2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Deconv2/strided_slice/stack_2?
Deconv2/strided_sliceStridedSliceDeconv2/Shape:output:0$Deconv2/strided_slice/stack:output:0&Deconv2/strided_slice/stack_1:output:0&Deconv2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Deconv2/strided_slicee
Deconv2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Deconv2/stack/1e
Deconv2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Deconv2/stack/2d
Deconv2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
Deconv2/stack/3?
Deconv2/stackPackDeconv2/strided_slice:output:0Deconv2/stack/1:output:0Deconv2/stack/2:output:0Deconv2/stack/3:output:0*
N*
T0*
_output_shapes
:2
Deconv2/stack?
Deconv2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Deconv2/strided_slice_1/stack?
Deconv2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
Deconv2/strided_slice_1/stack_1?
Deconv2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
Deconv2/strided_slice_1/stack_2?
Deconv2/strided_slice_1StridedSliceDeconv2/stack:output:0&Deconv2/strided_slice_1/stack:output:0(Deconv2/strided_slice_1/stack_1:output:0(Deconv2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Deconv2/strided_slice_1?
'Deconv2/conv2d_transpose/ReadVariableOpReadVariableOp0deconv2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		 *
dtype02)
'Deconv2/conv2d_transpose/ReadVariableOp?
Deconv2/conv2d_transposeConv2DBackpropInputDeconv2/stack:output:0/Deconv2/conv2d_transpose/ReadVariableOp:value:0Deconv1/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Deconv2/conv2d_transpose?
Deconv2/BiasAdd/ReadVariableOpReadVariableOp'deconv2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
Deconv2/BiasAdd/ReadVariableOp?
Deconv2/BiasAddBiasAdd!Deconv2/conv2d_transpose:output:0&Deconv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
Deconv2/BiasAdd?
tf.math.tanh_1/TanhTanhDeconv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
tf.math.tanh_1/Tanh?
IdentityIdentitytf.math.tanh_1/Tanh:y:0^Conv1/BiasAdd/ReadVariableOp^Conv1/Conv2D/ReadVariableOp^Conv2/BiasAdd/ReadVariableOp^Conv2/Conv2D/ReadVariableOp^Deconv1/BiasAdd/ReadVariableOp(^Deconv1/conv2d_transpose/ReadVariableOp^Deconv2/BiasAdd/ReadVariableOp(^Deconv2/conv2d_transpose/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2<
Conv1/BiasAdd/ReadVariableOpConv1/BiasAdd/ReadVariableOp2:
Conv1/Conv2D/ReadVariableOpConv1/Conv2D/ReadVariableOp2<
Conv2/BiasAdd/ReadVariableOpConv2/BiasAdd/ReadVariableOp2:
Conv2/Conv2D/ReadVariableOpConv2/Conv2D/ReadVariableOp2@
Deconv1/BiasAdd/ReadVariableOpDeconv1/BiasAdd/ReadVariableOp2R
'Deconv1/conv2d_transpose/ReadVariableOp'Deconv1/conv2d_transpose/ReadVariableOp2@
Deconv2/BiasAdd/ReadVariableOpDeconv2/BiasAdd/ReadVariableOp2R
'Deconv2/conv2d_transpose/ReadVariableOp'Deconv2/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
(__inference_fcn_model_layer_call_fn_5828

inputs!
unknown:		 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5:		 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_fcn_model_layer_call_and_return_conditional_losses_54912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
$__inference_Conv1_layer_call_fn_5858

inputs!
unknown:		 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Conv1_layer_call_and_return_conditional_losses_54562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
$__inference_Conv2_layer_call_fn_5878

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_Conv2_layer_call_and_return_conditional_losses_54732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
?__inference_Conv2_layer_call_and_return_conditional_losses_5889

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
&__inference_Deconv2_layer_call_fn_5438

inputs!
unknown:		 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_Deconv2_layer_call_and_return_conditional_losses_54282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K

InputLayer=
serving_default_InputLayer:0???????????L
tf.math.tanh_1:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?C
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
>_default_save_signature
*?&call_and_return_all_conditional_losses
@__call__"?@
_tf_keras_network?@{"name": "fcn_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "fcn_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "InputLayer"}, "name": "InputLayer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "Conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1", "inbound_nodes": [[["InputLayer", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "Conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv2", "inbound_nodes": [[["Conv1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Deconv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Deconv1", "inbound_nodes": [[["Conv2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "Deconv2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Deconv2", "inbound_nodes": [[["Deconv1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.tanh_1", "trainable": true, "dtype": "float32", "function": "math.tanh"}, "name": "tf.math.tanh_1", "inbound_nodes": [["Deconv2", 0, 0, {"name": null}]]}], "input_layers": [["InputLayer", 0, 0]], "output_layers": [["tf.math.tanh_1", 0, 0]]}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "float32", "InputLayer"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "fcn_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "InputLayer"}, "name": "InputLayer", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "Conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1", "inbound_nodes": [[["InputLayer", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "Conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv2", "inbound_nodes": [[["Conv1", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv2DTranspose", "config": {"name": "Deconv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Deconv1", "inbound_nodes": [[["Conv2", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Conv2DTranspose", "config": {"name": "Deconv2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "Deconv2", "inbound_nodes": [[["Deconv1", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.tanh_1", "trainable": true, "dtype": "float32", "function": "math.tanh"}, "name": "tf.math.tanh_1", "inbound_nodes": [["Deconv2", 0, 0, {"name": null}]], "shared_object_id": 13}], "input_layers": [["InputLayer", 0, 0]], "output_layers": [["tf.math.tanh_1", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "InputLayer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "InputLayer"}}
?


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "Conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["InputLayer", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}}
?


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
C__call__
*D&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "Conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Conv1", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 32]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
E__call__
*F&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "Deconv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "Deconv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["Conv2", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 64]}}
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
G__call__
*H&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "Deconv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "Deconv2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["Deconv1", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 32]}}
?
$	keras_api"?
_tf_keras_layer?{"name": "tf.math.tanh_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.tanh_1", "trainable": true, "dtype": "float32", "function": "math.tanh"}, "inbound_nodes": [["Deconv2", 0, 0, {"name": null}]], "shared_object_id": 13}
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
%non_trainable_variables
&layer_regularization_losses
'metrics
(layer_metrics
	variables
trainable_variables
	regularization_losses

)layers
@__call__
>_default_save_signature
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
,
Iserving_default"
signature_map
&:$		 2Conv1/kernel
: 2
Conv1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
*non_trainable_variables
+layer_regularization_losses
,metrics
-layer_metrics
	variables
trainable_variables
regularization_losses

.layers
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
&:$ @2Conv2/kernel
:@2
Conv2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
/non_trainable_variables
0layer_regularization_losses
1metrics
2layer_metrics
	variables
trainable_variables
regularization_losses

3layers
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
(:& @2Deconv1/kernel
: 2Deconv1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
4non_trainable_variables
5layer_regularization_losses
6metrics
7layer_metrics
	variables
trainable_variables
regularization_losses

8layers
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
(:&		 2Deconv2/kernel
:2Deconv2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
9non_trainable_variables
:layer_regularization_losses
;metrics
<layer_metrics
 	variables
!trainable_variables
"regularization_losses

=layers
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
__inference__wrapped_model_5349?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+

InputLayer???????????
?2?
C__inference_fcn_model_layer_call_and_return_conditional_losses_5749
C__inference_fcn_model_layer_call_and_return_conditional_losses_5807
C__inference_fcn_model_layer_call_and_return_conditional_losses_5643
C__inference_fcn_model_layer_call_and_return_conditional_losses_5668?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_fcn_model_layer_call_fn_5510
(__inference_fcn_model_layer_call_fn_5828
(__inference_fcn_model_layer_call_fn_5849
(__inference_fcn_model_layer_call_fn_5618?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_Conv1_layer_call_fn_5858?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_Conv1_layer_call_and_return_conditional_losses_5869?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_Conv2_layer_call_fn_5878?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_Conv2_layer_call_and_return_conditional_losses_5889?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_Deconv1_layer_call_fn_5394?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
A__inference_Deconv1_layer_call_and_return_conditional_losses_5384?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
&__inference_Deconv2_layer_call_fn_5438?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
A__inference_Deconv2_layer_call_and_return_conditional_losses_5428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?B?
"__inference_signature_wrapper_5691
InputLayer"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
?__inference_Conv1_layer_call_and_return_conditional_losses_5869p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
$__inference_Conv1_layer_call_fn_5858c9?6
/?,
*?'
inputs???????????
? ""???????????? ?
?__inference_Conv2_layer_call_and_return_conditional_losses_5889p9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????@
? ?
$__inference_Conv2_layer_call_fn_5878c9?6
/?,
*?'
inputs??????????? 
? ""????????????@?
A__inference_Deconv1_layer_call_and_return_conditional_losses_5384?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
&__inference_Deconv1_layer_call_fn_5394?I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
A__inference_Deconv2_layer_call_and_return_conditional_losses_5428?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
&__inference_Deconv2_layer_call_fn_5438?I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
__inference__wrapped_model_5349?=?:
3?0
.?+

InputLayer???????????
? "I?F
D
tf.math.tanh_12?/
tf.math.tanh_1????????????
C__inference_fcn_model_layer_call_and_return_conditional_losses_5643?E?B
;?8
.?+

InputLayer???????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_fcn_model_layer_call_and_return_conditional_losses_5668?E?B
;?8
.?+

InputLayer???????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_fcn_model_layer_call_and_return_conditional_losses_5749~A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
C__inference_fcn_model_layer_call_and_return_conditional_losses_5807~A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
(__inference_fcn_model_layer_call_fn_5510?E?B
;?8
.?+

InputLayer???????????
p 

 
? "2?/+????????????????????????????
(__inference_fcn_model_layer_call_fn_5618?E?B
;?8
.?+

InputLayer???????????
p

 
? "2?/+????????????????????????????
(__inference_fcn_model_layer_call_fn_5828?A?>
7?4
*?'
inputs???????????
p 

 
? "2?/+????????????????????????????
(__inference_fcn_model_layer_call_fn_5849?A?>
7?4
*?'
inputs???????????
p

 
? "2?/+????????????????????????????
"__inference_signature_wrapper_5691?K?H
? 
A?>
<

InputLayer.?+

InputLayer???????????"I?F
D
tf.math.tanh_12?/
tf.math.tanh_1???????????