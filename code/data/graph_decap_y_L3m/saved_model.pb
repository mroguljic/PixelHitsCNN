�
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
@
Softplus
features"T
activations"T"
Ttype:
2
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.12v2.13.0-17-gf841394b1b78��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
�
Adam/v/dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_43/bias
y
(Adam/v/dense_43/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_43/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_43/bias
y
(Adam/m/dense_43/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_43/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/v/dense_43/kernel
�
*Adam/v/dense_43/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_43/kernel*
_output_shapes

:`*
dtype0
�
Adam/m/dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/m/dense_43/kernel
�
*Adam/m/dense_43/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_43/kernel*
_output_shapes

:`*
dtype0
�
Adam/v/dense_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_47/bias
y
(Adam/v/dense_47/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_47/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_47/bias
y
(Adam/m/dense_47/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_47/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/v/dense_47/kernel
�
*Adam/v/dense_47/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_47/kernel*
_output_shapes

:`*
dtype0
�
Adam/m/dense_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/m/dense_47/kernel
�
*Adam/m/dense_47/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_47/kernel*
_output_shapes

:`*
dtype0
�
Adam/v/dense_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/v/dense_42/bias
y
(Adam/v/dense_42/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_42/bias*
_output_shapes
:`*
dtype0
�
Adam/m/dense_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/m/dense_42/bias
y
(Adam/m/dense_42/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_42/bias*
_output_shapes
:`*
dtype0
�
Adam/v/dense_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/v/dense_42/kernel
�
*Adam/v/dense_42/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_42/kernel*
_output_shapes

:`*
dtype0
�
Adam/m/dense_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/m/dense_42/kernel
�
*Adam/m/dense_42/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_42/kernel*
_output_shapes

:`*
dtype0
�
Adam/v/dense_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/v/dense_46/bias
y
(Adam/v/dense_46/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_46/bias*
_output_shapes
:`*
dtype0
�
Adam/m/dense_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/m/dense_46/bias
y
(Adam/m/dense_46/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_46/bias*
_output_shapes
:`*
dtype0
�
Adam/v/dense_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/v/dense_46/kernel
�
*Adam/v/dense_46/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_46/kernel*
_output_shapes

:`*
dtype0
�
Adam/m/dense_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/m/dense_46/kernel
�
*Adam/m/dense_46/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_46/kernel*
_output_shapes

:`*
dtype0
�
"Adam/v/batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/v/batch_normalization_10/beta
�
6Adam/v/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_10/beta*
_output_shapes
:`*
dtype0
�
"Adam/m/batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/m/batch_normalization_10/beta
�
6Adam/m/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_10/beta*
_output_shapes
:`*
dtype0
�
#Adam/v/batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*4
shared_name%#Adam/v/batch_normalization_10/gamma
�
7Adam/v/batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_10/gamma*
_output_shapes
:`*
dtype0
�
#Adam/m/batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*4
shared_name%#Adam/m/batch_normalization_10/gamma
�
7Adam/m/batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_10/gamma*
_output_shapes
:`*
dtype0
�
"Adam/v/batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/v/batch_normalization_11/beta
�
6Adam/v/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_11/beta*
_output_shapes
:`*
dtype0
�
"Adam/m/batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"Adam/m/batch_normalization_11/beta
�
6Adam/m/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_11/beta*
_output_shapes
:`*
dtype0
�
#Adam/v/batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*4
shared_name%#Adam/v/batch_normalization_11/gamma
�
7Adam/v/batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_11/gamma*
_output_shapes
:`*
dtype0
�
#Adam/m/batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*4
shared_name%#Adam/m/batch_normalization_11/gamma
�
7Adam/m/batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_11/gamma*
_output_shapes
:`*
dtype0
�
Adam/v/dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/v/dense_41/bias
y
(Adam/v/dense_41/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_41/bias*
_output_shapes
:`*
dtype0
�
Adam/m/dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/m/dense_41/bias
y
(Adam/m/dense_41/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_41/bias*
_output_shapes
:`*
dtype0
�
Adam/v/dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�`*'
shared_nameAdam/v/dense_41/kernel
�
*Adam/v/dense_41/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_41/kernel*
_output_shapes
:	�`*
dtype0
�
Adam/m/dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�`*'
shared_nameAdam/m/dense_41/kernel
�
*Adam/m/dense_41/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_41/kernel*
_output_shapes
:	�`*
dtype0
�
Adam/v/dense_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/v/dense_45/bias
y
(Adam/v/dense_45/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_45/bias*
_output_shapes
:`*
dtype0
�
Adam/m/dense_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/m/dense_45/bias
y
(Adam/m/dense_45/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_45/bias*
_output_shapes
:`*
dtype0
�
Adam/v/dense_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�`*'
shared_nameAdam/v/dense_45/kernel
�
*Adam/v/dense_45/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_45/kernel*
_output_shapes
:	�`*
dtype0
�
Adam/m/dense_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�`*'
shared_nameAdam/m/dense_45/kernel
�
*Adam/m/dense_45/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_45/kernel*
_output_shapes
:	�`*
dtype0
�
Adam/v/dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_40/bias
z
(Adam/v/dense_40/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_40/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_40/bias
z
(Adam/m/dense_40/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_40/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/v/dense_40/kernel
�
*Adam/v/dense_40/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_40/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/m/dense_40/kernel
�
*Adam/m/dense_40/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_40/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_44/bias
z
(Adam/v/dense_44/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_44/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_44/bias
z
(Adam/m/dense_44/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_44/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/v/dense_44/kernel
�
*Adam/v/dense_44/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_44/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/m/dense_44/kernel
�
*Adam/m/dense_44/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_44/kernel*
_output_shapes
:	�*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_43/bias
k
!dense_43/bias/Read/ReadVariableOpReadVariableOpdense_43/bias*
_output_shapes
:*
dtype0
z
dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`* 
shared_namedense_43/kernel
s
#dense_43/kernel/Read/ReadVariableOpReadVariableOpdense_43/kernel*
_output_shapes

:`*
dtype0
r
dense_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_47/bias
k
!dense_47/bias/Read/ReadVariableOpReadVariableOpdense_47/bias*
_output_shapes
:*
dtype0
z
dense_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`* 
shared_namedense_47/kernel
s
#dense_47/kernel/Read/ReadVariableOpReadVariableOpdense_47/kernel*
_output_shapes

:`*
dtype0
r
dense_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_namedense_42/bias
k
!dense_42/bias/Read/ReadVariableOpReadVariableOpdense_42/bias*
_output_shapes
:`*
dtype0
z
dense_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`* 
shared_namedense_42/kernel
s
#dense_42/kernel/Read/ReadVariableOpReadVariableOpdense_42/kernel*
_output_shapes

:`*
dtype0
r
dense_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_namedense_46/bias
k
!dense_46/bias/Read/ReadVariableOpReadVariableOpdense_46/bias*
_output_shapes
:`*
dtype0
z
dense_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`* 
shared_namedense_46/kernel
s
#dense_46/kernel/Read/ReadVariableOpReadVariableOpdense_46/kernel*
_output_shapes

:`*
dtype0
�
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*7
shared_name(&batch_normalization_10/moving_variance
�
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:`*
dtype0
�
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"batch_normalization_10/moving_mean
�
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:`*
dtype0
�
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namebatch_normalization_10/beta
�
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:`*
dtype0
�
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*-
shared_namebatch_normalization_10/gamma
�
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:`*
dtype0
�
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*7
shared_name(&batch_normalization_11/moving_variance
�
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
:`*
dtype0
�
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"batch_normalization_11/moving_mean
�
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:`*
dtype0
�
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namebatch_normalization_11/beta
�
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:`*
dtype0
�
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*-
shared_namebatch_normalization_11/gamma
�
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
:`*
dtype0
r
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_namedense_41/bias
k
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
_output_shapes
:`*
dtype0
{
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�`* 
shared_namedense_41/kernel
t
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel*
_output_shapes
:	�`*
dtype0
r
dense_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_namedense_45/bias
k
!dense_45/bias/Read/ReadVariableOpReadVariableOpdense_45/bias*
_output_shapes
:`*
dtype0
{
dense_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�`* 
shared_namedense_45/kernel
t
#dense_45/kernel/Read/ReadVariableOpReadVariableOpdense_45/kernel*
_output_shapes
:	�`*
dtype0
s
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_40/bias
l
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes	
:�*
dtype0
{
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_40/kernel
t
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*
_output_shapes
:	�*
dtype0
s
dense_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_44/bias
l
!dense_44/bias/Read/ReadVariableOpReadVariableOpdense_44/bias*
_output_shapes	
:�*
dtype0
{
dense_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_44/kernel
t
#dense_44/kernel/Read/ReadVariableOpReadVariableOpdense_44/kernel*
_output_shapes
:	�*
dtype0
y
serving_default_anglesPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
serving_default_cluster_chargePlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
"serving_default_pixel_projection_yPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_anglesserving_default_cluster_charge"serving_default_pixel_projection_ydense_44/kerneldense_44/biasdense_40/kerneldense_40/biasdense_45/kerneldense_45/biasdense_41/kerneldense_41/bias&batch_normalization_11/moving_variancebatch_normalization_11/gamma"batch_normalization_11/moving_meanbatch_normalization_11/betadense_46/kerneldense_46/bias&batch_normalization_10/moving_variancebatch_normalization_10/gamma"batch_normalization_10/moving_meanbatch_normalization_10/betadense_42/kerneldense_42/biasdense_47/kerneldense_47/biasdense_43/kerneldense_43/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *.
f)R'
%__inference_signature_wrapper_1804531

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ů
value��B�� B��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
layer-15
layer_with_weights-6
layer-16
layer-17
layer_with_weights-7
layer-18
layer-19
layer-20
layer_with_weights-8
layer-21
layer_with_weights-9
layer-22
layer-23
layer-24
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!	optimizer
"
signatures*
* 
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
* 
* 
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses* 
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias*
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses* 
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses* 
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias*
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias*
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses* 
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance*
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
xaxis
	ygamma
zbeta
{moving_mean
|moving_variance*
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*

�	keras_api* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
50
61
=2
>3
Q4
R5
Y6
Z7
n8
o9
p10
q11
y12
z13
{14
|15
�16
�17
�18
�19
�20
�21
�22
�23*
�
50
61
=2
>3
Q4
R5
Y6
Z7
n8
o9
y10
z11
�12
�13
�14
�15
�16
�17
�18
�19*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

50
61*

50
61*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_44/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_44/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

=0
>1*

=0
>1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_40/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_40/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

Q0
R1*

Q0
R1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_45/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_45/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_41/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_41/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
 
n0
o1
p2
q3*

n0
o1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
y0
z1
{2
|3*

y0
z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_46/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_46/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_42/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_42/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_47/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_47/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_43/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_43/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
 
p0
q1
{2
|3*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24*

�0
�1
�2*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19*
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11
�trace_12
�trace_13
�trace_14
�trace_15
�trace_16
�trace_17
�trace_18
�trace_19* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

p0
q1*
* 
* 
* 
* 
* 
* 
* 
* 

{0
|1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/dense_44/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_44/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_44/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_44/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_40/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_40/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_40/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_40/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_45/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_45/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_45/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_45/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_41/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_41/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_41/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_41/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_11/gamma2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_11/gamma2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_11/beta2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_11/beta2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_10/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_10/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_10/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_10/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_46/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_46/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_46/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_46/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_42/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_42/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_42/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_42/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_47/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_47/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_47/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_47/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_43/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_43/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_43/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_43/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_44/kerneldense_44/biasdense_40/kerneldense_40/biasdense_45/kerneldense_45/biasdense_41/kerneldense_41/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancebatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_variancedense_46/kerneldense_46/biasdense_42/kerneldense_42/biasdense_47/kerneldense_47/biasdense_43/kerneldense_43/bias	iterationlearning_rateAdam/m/dense_44/kernelAdam/v/dense_44/kernelAdam/m/dense_44/biasAdam/v/dense_44/biasAdam/m/dense_40/kernelAdam/v/dense_40/kernelAdam/m/dense_40/biasAdam/v/dense_40/biasAdam/m/dense_45/kernelAdam/v/dense_45/kernelAdam/m/dense_45/biasAdam/v/dense_45/biasAdam/m/dense_41/kernelAdam/v/dense_41/kernelAdam/m/dense_41/biasAdam/v/dense_41/bias#Adam/m/batch_normalization_11/gamma#Adam/v/batch_normalization_11/gamma"Adam/m/batch_normalization_11/beta"Adam/v/batch_normalization_11/beta#Adam/m/batch_normalization_10/gamma#Adam/v/batch_normalization_10/gamma"Adam/m/batch_normalization_10/beta"Adam/v/batch_normalization_10/betaAdam/m/dense_46/kernelAdam/v/dense_46/kernelAdam/m/dense_46/biasAdam/v/dense_46/biasAdam/m/dense_42/kernelAdam/v/dense_42/kernelAdam/m/dense_42/biasAdam/v/dense_42/biasAdam/m/dense_47/kernelAdam/v/dense_47/kernelAdam/m/dense_47/biasAdam/v/dense_47/biasAdam/m/dense_43/kernelAdam/v/dense_43/kernelAdam/m/dense_43/biasAdam/v/dense_43/biastotal_2count_2total_1count_1totalcountConst*U
TinN
L2J*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *)
f$R"
 __inference__traced_save_1805557
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_44/kerneldense_44/biasdense_40/kerneldense_40/biasdense_45/kerneldense_45/biasdense_41/kerneldense_41/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancebatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_variancedense_46/kerneldense_46/biasdense_42/kerneldense_42/biasdense_47/kerneldense_47/biasdense_43/kerneldense_43/bias	iterationlearning_rateAdam/m/dense_44/kernelAdam/v/dense_44/kernelAdam/m/dense_44/biasAdam/v/dense_44/biasAdam/m/dense_40/kernelAdam/v/dense_40/kernelAdam/m/dense_40/biasAdam/v/dense_40/biasAdam/m/dense_45/kernelAdam/v/dense_45/kernelAdam/m/dense_45/biasAdam/v/dense_45/biasAdam/m/dense_41/kernelAdam/v/dense_41/kernelAdam/m/dense_41/biasAdam/v/dense_41/bias#Adam/m/batch_normalization_11/gamma#Adam/v/batch_normalization_11/gamma"Adam/m/batch_normalization_11/beta"Adam/v/batch_normalization_11/beta#Adam/m/batch_normalization_10/gamma#Adam/v/batch_normalization_10/gamma"Adam/m/batch_normalization_10/beta"Adam/v/batch_normalization_10/betaAdam/m/dense_46/kernelAdam/v/dense_46/kernelAdam/m/dense_46/biasAdam/v/dense_46/biasAdam/m/dense_42/kernelAdam/v/dense_42/kernelAdam/m/dense_42/biasAdam/v/dense_42/biasAdam/m/dense_47/kernelAdam/v/dense_47/kernelAdam/m/dense_47/biasAdam/v/dense_47/biasAdam/m/dense_43/kernelAdam/v/dense_43/kernelAdam/m/dense_43/biasAdam/v/dense_43/biastotal_2count_2total_1count_1totalcount*T
TinM
K2I*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *,
f'R%
#__inference__traced_restore_1805782��
�
L
$__inference__update_step_xla_1804576
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:`: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:`
"
_user_specified_name
gradient
�&
�
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1804833

inputs5
'assignmovingavg_readvariableop_resource:`7
)assignmovingavg_1_readvariableop_resource:`3
%batchnorm_mul_readvariableop_resource:`/
!batchnorm_readvariableop_resource:`
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:`*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:`�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������`l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:`*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:`*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:`x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:`�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:`*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:`~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:`�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:`P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:`~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������`h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:`v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:`r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������`b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�	
�
E__inference_dense_46_layer_call_and_return_conditional_losses_1804031

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������`S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
o
C__inference_add_11_layer_call_and_return_conditional_losses_1805037
inputs_0
inputs_1
identityR
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:���������`O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������`:���������`:QM
'
_output_shapes
:���������`
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������`
"
_user_specified_name
inputs_0
�
L
0__inference_leaky_re_lu_21_layer_call_fn_1804768

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1804007`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
*__inference_dense_44_layer_call_fn_1804666

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_44_layer_call_and_return_conditional_losses_1803925p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1804662:'#
!
_user_specified_name	1804660:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1804913

inputs5
'assignmovingavg_readvariableop_resource:`7
)assignmovingavg_1_readvariableop_resource:`3
%batchnorm_mul_readvariableop_resource:`/
!batchnorm_readvariableop_resource:`
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:`*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:`�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������`l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:`*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:`*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:`x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:`�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:`*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:`~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:`�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:`P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:`~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������`h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:`v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:`r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������`b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
e
,__inference_dropout_10_layer_call_fn_1804984

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_1804064o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�	
�
E__inference_dense_40_layer_call_and_return_conditional_losses_1803946

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
w
K__inference_concatenate_11_layer_call_and_return_conditional_losses_1805101
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
T
(__inference_add_10_layer_call_fn_1805043
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_add_10_layer_call_and_return_conditional_losses_1804102`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������`:���������`:QM
'
_output_shapes
:���������`
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������`
"
_user_specified_name
inputs_0
�
g
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1804773

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������`*
alpha%
�#<_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
*__inference_dense_40_layer_call_fn_1804685

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_1803946p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1804681:'#
!
_user_specified_name	1804679:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

f
G__inference_dropout_10_layer_call_and_return_conditional_losses_1804064

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������`Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������`*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������`T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������`a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�	
�
E__inference_dense_42_layer_call_and_return_conditional_losses_1805025

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������`S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
L
$__inference__update_step_xla_1804631
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
�	
�
E__inference_dense_43_layer_call_and_return_conditional_losses_1804113

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1803869

inputs/
!batchnorm_readvariableop_resource:`3
%batchnorm_mul_readvariableop_resource:`1
#batchnorm_readvariableop_1_resource:`1
#batchnorm_readvariableop_2_resource:`
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:`P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:`~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������`z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:`z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:`*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:`r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������`b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
*__inference_dense_47_layer_call_fn_1805058

inputs
unknown:`
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_47_layer_call_and_return_conditional_losses_1804091o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1805054:'#
!
_user_specified_name	1805052:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
)__inference_model_5_layer_call_fn_1804331
pixel_projection_y

angles
cluster_charge
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�`
	unknown_4:`
	unknown_5:	�`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:`

unknown_10:`

unknown_11:`

unknown_12:`

unknown_13:`

unknown_14:`

unknown_15:`

unknown_16:`

unknown_17:`

unknown_18:`

unknown_19:`

unknown_20:

unknown_21:`

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallpixel_projection_yanglescluster_chargeunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_model_5_layer_call_and_return_conditional_losses_1804221o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:���������:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1804327:'#
!
_user_specified_name	1804325:'#
!
_user_specified_name	1804323:'#
!
_user_specified_name	1804321:'#
!
_user_specified_name	1804319:'#
!
_user_specified_name	1804317:'#
!
_user_specified_name	1804315:'#
!
_user_specified_name	1804313:'#
!
_user_specified_name	1804311:'#
!
_user_specified_name	1804309:'#
!
_user_specified_name	1804307:'#
!
_user_specified_name	1804305:'#
!
_user_specified_name	1804303:'#
!
_user_specified_name	1804301:'#
!
_user_specified_name	1804299:'#
!
_user_specified_name	1804297:'
#
!
_user_specified_name	1804295:'	#
!
_user_specified_name	1804293:'#
!
_user_specified_name	1804291:'#
!
_user_specified_name	1804289:'#
!
_user_specified_name	1804287:'#
!
_user_specified_name	1804285:'#
!
_user_specified_name	1804283:'#
!
_user_specified_name	1804281:WS
'
_output_shapes
:���������
(
_user_specified_namecluster_charge:OK
'
_output_shapes
:���������
 
_user_specified_nameangles:_ [
+
_output_shapes
:���������
,
_user_specified_namepixel_projection_y
�	
�
E__inference_dense_43_layer_call_and_return_conditional_losses_1805088

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
Q
$__inference__update_step_xla_1804536
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	�
"
_user_specified_name
gradient
�
�
%__inference_signature_wrapper_1804531

angles
cluster_charge
pixel_projection_y
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�`
	unknown_4:`
	unknown_5:	�`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:`

unknown_10:`

unknown_11:`

unknown_12:`

unknown_13:`

unknown_14:`

unknown_15:`

unknown_16:`

unknown_17:`

unknown_18:`

unknown_19:`

unknown_20:

unknown_21:`

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallpixel_projection_yanglescluster_chargeunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *+
f&R$
"__inference__wrapped_model_1803735o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:���������:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1804527:'#
!
_user_specified_name	1804525:'#
!
_user_specified_name	1804523:'#
!
_user_specified_name	1804521:'#
!
_user_specified_name	1804519:'#
!
_user_specified_name	1804517:'#
!
_user_specified_name	1804515:'#
!
_user_specified_name	1804513:'#
!
_user_specified_name	1804511:'#
!
_user_specified_name	1804509:'#
!
_user_specified_name	1804507:'#
!
_user_specified_name	1804505:'#
!
_user_specified_name	1804503:'#
!
_user_specified_name	1804501:'#
!
_user_specified_name	1804499:'#
!
_user_specified_name	1804497:'
#
!
_user_specified_name	1804495:'	#
!
_user_specified_name	1804493:'#
!
_user_specified_name	1804491:'#
!
_user_specified_name	1804489:'#
!
_user_specified_name	1804487:'#
!
_user_specified_name	1804485:'#
!
_user_specified_name	1804483:'#
!
_user_specified_name	1804481:_[
+
_output_shapes
:���������
,
_user_specified_namepixel_projection_y:WS
'
_output_shapes
:���������
(
_user_specified_namecluster_charge:O K
'
_output_shapes
:���������
 
_user_specified_nameangles
�
M
$__inference__update_step_xla_1804551
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�
g
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1804763

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������`*
alpha%
�#<_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
e
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804960

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������`[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
T
(__inference_add_11_layer_call_fn_1805031
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_add_11_layer_call_and_return_conditional_losses_1804051`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������`:���������`:QM
'
_output_shapes
:���������`
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������`
"
_user_specified_name
inputs_0
�
H
,__inference_dropout_10_layer_call_fn_1804989

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_1804197`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
ޝ
�B
 __inference__traced_save_1805557
file_prefix9
&read_disablecopyonread_dense_44_kernel:	�5
&read_1_disablecopyonread_dense_44_bias:	�;
(read_2_disablecopyonread_dense_40_kernel:	�5
&read_3_disablecopyonread_dense_40_bias:	�;
(read_4_disablecopyonread_dense_45_kernel:	�`4
&read_5_disablecopyonread_dense_45_bias:`;
(read_6_disablecopyonread_dense_41_kernel:	�`4
&read_7_disablecopyonread_dense_41_bias:`C
5read_8_disablecopyonread_batch_normalization_11_gamma:`B
4read_9_disablecopyonread_batch_normalization_11_beta:`J
<read_10_disablecopyonread_batch_normalization_11_moving_mean:`N
@read_11_disablecopyonread_batch_normalization_11_moving_variance:`D
6read_12_disablecopyonread_batch_normalization_10_gamma:`C
5read_13_disablecopyonread_batch_normalization_10_beta:`J
<read_14_disablecopyonread_batch_normalization_10_moving_mean:`N
@read_15_disablecopyonread_batch_normalization_10_moving_variance:`;
)read_16_disablecopyonread_dense_46_kernel:`5
'read_17_disablecopyonread_dense_46_bias:`;
)read_18_disablecopyonread_dense_42_kernel:`5
'read_19_disablecopyonread_dense_42_bias:`;
)read_20_disablecopyonread_dense_47_kernel:`5
'read_21_disablecopyonread_dense_47_bias:;
)read_22_disablecopyonread_dense_43_kernel:`5
'read_23_disablecopyonread_dense_43_bias:-
#read_24_disablecopyonread_iteration:	 1
'read_25_disablecopyonread_learning_rate: C
0read_26_disablecopyonread_adam_m_dense_44_kernel:	�C
0read_27_disablecopyonread_adam_v_dense_44_kernel:	�=
.read_28_disablecopyonread_adam_m_dense_44_bias:	�=
.read_29_disablecopyonread_adam_v_dense_44_bias:	�C
0read_30_disablecopyonread_adam_m_dense_40_kernel:	�C
0read_31_disablecopyonread_adam_v_dense_40_kernel:	�=
.read_32_disablecopyonread_adam_m_dense_40_bias:	�=
.read_33_disablecopyonread_adam_v_dense_40_bias:	�C
0read_34_disablecopyonread_adam_m_dense_45_kernel:	�`C
0read_35_disablecopyonread_adam_v_dense_45_kernel:	�`<
.read_36_disablecopyonread_adam_m_dense_45_bias:`<
.read_37_disablecopyonread_adam_v_dense_45_bias:`C
0read_38_disablecopyonread_adam_m_dense_41_kernel:	�`C
0read_39_disablecopyonread_adam_v_dense_41_kernel:	�`<
.read_40_disablecopyonread_adam_m_dense_41_bias:`<
.read_41_disablecopyonread_adam_v_dense_41_bias:`K
=read_42_disablecopyonread_adam_m_batch_normalization_11_gamma:`K
=read_43_disablecopyonread_adam_v_batch_normalization_11_gamma:`J
<read_44_disablecopyonread_adam_m_batch_normalization_11_beta:`J
<read_45_disablecopyonread_adam_v_batch_normalization_11_beta:`K
=read_46_disablecopyonread_adam_m_batch_normalization_10_gamma:`K
=read_47_disablecopyonread_adam_v_batch_normalization_10_gamma:`J
<read_48_disablecopyonread_adam_m_batch_normalization_10_beta:`J
<read_49_disablecopyonread_adam_v_batch_normalization_10_beta:`B
0read_50_disablecopyonread_adam_m_dense_46_kernel:`B
0read_51_disablecopyonread_adam_v_dense_46_kernel:`<
.read_52_disablecopyonread_adam_m_dense_46_bias:`<
.read_53_disablecopyonread_adam_v_dense_46_bias:`B
0read_54_disablecopyonread_adam_m_dense_42_kernel:`B
0read_55_disablecopyonread_adam_v_dense_42_kernel:`<
.read_56_disablecopyonread_adam_m_dense_42_bias:`<
.read_57_disablecopyonread_adam_v_dense_42_bias:`B
0read_58_disablecopyonread_adam_m_dense_47_kernel:`B
0read_59_disablecopyonread_adam_v_dense_47_kernel:`<
.read_60_disablecopyonread_adam_m_dense_47_bias:<
.read_61_disablecopyonread_adam_v_dense_47_bias:B
0read_62_disablecopyonread_adam_m_dense_43_kernel:`B
0read_63_disablecopyonread_adam_v_dense_43_kernel:`<
.read_64_disablecopyonread_adam_m_dense_43_bias:<
.read_65_disablecopyonread_adam_v_dense_43_bias:+
!read_66_disablecopyonread_total_2: +
!read_67_disablecopyonread_count_2: +
!read_68_disablecopyonread_total_1: +
!read_69_disablecopyonread_count_1: )
read_70_disablecopyonread_total: )
read_71_disablecopyonread_count: 
savev2_const
identity_145��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_44_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_44_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_44_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_44_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_40_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_40_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_40_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_40_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_45_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_45_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�`*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�`d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	�`z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_45_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_45_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:`|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_dense_41_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_dense_41_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�`*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�`f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	�`z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_dense_41_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_dense_41_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_8/DisableCopyOnReadDisableCopyOnRead5read_8_disablecopyonread_batch_normalization_11_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp5read_8_disablecopyonread_batch_normalization_11_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_9/DisableCopyOnReadDisableCopyOnRead4read_9_disablecopyonread_batch_normalization_11_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp4read_9_disablecopyonread_batch_normalization_11_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_10/DisableCopyOnReadDisableCopyOnRead<read_10_disablecopyonread_batch_normalization_11_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp<read_10_disablecopyonread_batch_normalization_11_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_11/DisableCopyOnReadDisableCopyOnRead@read_11_disablecopyonread_batch_normalization_11_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp@read_11_disablecopyonread_batch_normalization_11_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_12/DisableCopyOnReadDisableCopyOnRead6read_12_disablecopyonread_batch_normalization_10_gamma"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp6read_12_disablecopyonread_batch_normalization_10_gamma^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_13/DisableCopyOnReadDisableCopyOnRead5read_13_disablecopyonread_batch_normalization_10_beta"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp5read_13_disablecopyonread_batch_normalization_10_beta^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_14/DisableCopyOnReadDisableCopyOnRead<read_14_disablecopyonread_batch_normalization_10_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp<read_14_disablecopyonread_batch_normalization_10_moving_mean^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_15/DisableCopyOnReadDisableCopyOnRead@read_15_disablecopyonread_batch_normalization_10_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp@read_15_disablecopyonread_batch_normalization_10_moving_variance^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:`~
Read_16/DisableCopyOnReadDisableCopyOnRead)read_16_disablecopyonread_dense_46_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp)read_16_disablecopyonread_dense_46_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:`|
Read_17/DisableCopyOnReadDisableCopyOnRead'read_17_disablecopyonread_dense_46_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp'read_17_disablecopyonread_dense_46_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:`~
Read_18/DisableCopyOnReadDisableCopyOnRead)read_18_disablecopyonread_dense_42_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp)read_18_disablecopyonread_dense_42_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:`|
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_dense_42_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_dense_42_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:`~
Read_20/DisableCopyOnReadDisableCopyOnRead)read_20_disablecopyonread_dense_47_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp)read_20_disablecopyonread_dense_47_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:`|
Read_21/DisableCopyOnReadDisableCopyOnRead'read_21_disablecopyonread_dense_47_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp'read_21_disablecopyonread_dense_47_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_22/DisableCopyOnReadDisableCopyOnRead)read_22_disablecopyonread_dense_43_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp)read_22_disablecopyonread_dense_43_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:`|
Read_23/DisableCopyOnReadDisableCopyOnRead'read_23_disablecopyonread_dense_43_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp'read_23_disablecopyonread_dense_43_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_24/DisableCopyOnReadDisableCopyOnRead#read_24_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp#read_24_disablecopyonread_iteration^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_25/DisableCopyOnReadDisableCopyOnRead'read_25_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp'read_25_disablecopyonread_learning_rate^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_26/DisableCopyOnReadDisableCopyOnRead0read_26_disablecopyonread_adam_m_dense_44_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp0read_26_disablecopyonread_adam_m_dense_44_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_v_dense_44_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_v_dense_44_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_m_dense_44_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_m_dense_44_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_adam_v_dense_44_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_adam_v_dense_44_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_30/DisableCopyOnReadDisableCopyOnRead0read_30_disablecopyonread_adam_m_dense_40_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp0read_30_disablecopyonread_adam_m_dense_40_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_v_dense_40_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_v_dense_40_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_32/DisableCopyOnReadDisableCopyOnRead.read_32_disablecopyonread_adam_m_dense_40_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp.read_32_disablecopyonread_adam_m_dense_40_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_33/DisableCopyOnReadDisableCopyOnRead.read_33_disablecopyonread_adam_v_dense_40_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp.read_33_disablecopyonread_adam_v_dense_40_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_34/DisableCopyOnReadDisableCopyOnRead0read_34_disablecopyonread_adam_m_dense_45_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp0read_34_disablecopyonread_adam_m_dense_45_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�`*
dtype0p
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�`f
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:	�`�
Read_35/DisableCopyOnReadDisableCopyOnRead0read_35_disablecopyonread_adam_v_dense_45_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp0read_35_disablecopyonread_adam_v_dense_45_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�`*
dtype0p
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�`f
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:	�`�
Read_36/DisableCopyOnReadDisableCopyOnRead.read_36_disablecopyonread_adam_m_dense_45_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp.read_36_disablecopyonread_adam_m_dense_45_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_37/DisableCopyOnReadDisableCopyOnRead.read_37_disablecopyonread_adam_v_dense_45_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp.read_37_disablecopyonread_adam_v_dense_45_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_38/DisableCopyOnReadDisableCopyOnRead0read_38_disablecopyonread_adam_m_dense_41_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp0read_38_disablecopyonread_adam_m_dense_41_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�`*
dtype0p
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�`f
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:	�`�
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_v_dense_41_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_v_dense_41_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�`*
dtype0p
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�`f
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:	�`�
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_adam_m_dense_41_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_adam_m_dense_41_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_v_dense_41_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_v_dense_41_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_42/DisableCopyOnReadDisableCopyOnRead=read_42_disablecopyonread_adam_m_batch_normalization_11_gamma"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp=read_42_disablecopyonread_adam_m_batch_normalization_11_gamma^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_43/DisableCopyOnReadDisableCopyOnRead=read_43_disablecopyonread_adam_v_batch_normalization_11_gamma"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp=read_43_disablecopyonread_adam_v_batch_normalization_11_gamma^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_44/DisableCopyOnReadDisableCopyOnRead<read_44_disablecopyonread_adam_m_batch_normalization_11_beta"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp<read_44_disablecopyonread_adam_m_batch_normalization_11_beta^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_45/DisableCopyOnReadDisableCopyOnRead<read_45_disablecopyonread_adam_v_batch_normalization_11_beta"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp<read_45_disablecopyonread_adam_v_batch_normalization_11_beta^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_46/DisableCopyOnReadDisableCopyOnRead=read_46_disablecopyonread_adam_m_batch_normalization_10_gamma"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp=read_46_disablecopyonread_adam_m_batch_normalization_10_gamma^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_47/DisableCopyOnReadDisableCopyOnRead=read_47_disablecopyonread_adam_v_batch_normalization_10_gamma"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp=read_47_disablecopyonread_adam_v_batch_normalization_10_gamma^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_48/DisableCopyOnReadDisableCopyOnRead<read_48_disablecopyonread_adam_m_batch_normalization_10_beta"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp<read_48_disablecopyonread_adam_m_batch_normalization_10_beta^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_49/DisableCopyOnReadDisableCopyOnRead<read_49_disablecopyonread_adam_v_batch_normalization_10_beta"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp<read_49_disablecopyonread_adam_v_batch_normalization_10_beta^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_50/DisableCopyOnReadDisableCopyOnRead0read_50_disablecopyonread_adam_m_dense_46_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp0read_50_disablecopyonread_adam_m_dense_46_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0p
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`g
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_51/DisableCopyOnReadDisableCopyOnRead0read_51_disablecopyonread_adam_v_dense_46_kernel"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp0read_51_disablecopyonread_adam_v_dense_46_kernel^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0p
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`g
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_52/DisableCopyOnReadDisableCopyOnRead.read_52_disablecopyonread_adam_m_dense_46_bias"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp.read_52_disablecopyonread_adam_m_dense_46_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_53/DisableCopyOnReadDisableCopyOnRead.read_53_disablecopyonread_adam_v_dense_46_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp.read_53_disablecopyonread_adam_v_dense_46_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_54/DisableCopyOnReadDisableCopyOnRead0read_54_disablecopyonread_adam_m_dense_42_kernel"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp0read_54_disablecopyonread_adam_m_dense_42_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0p
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`g
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_55/DisableCopyOnReadDisableCopyOnRead0read_55_disablecopyonread_adam_v_dense_42_kernel"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp0read_55_disablecopyonread_adam_v_dense_42_kernel^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0p
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`g
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_56/DisableCopyOnReadDisableCopyOnRead.read_56_disablecopyonread_adam_m_dense_42_bias"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp.read_56_disablecopyonread_adam_m_dense_42_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0l
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_57/DisableCopyOnReadDisableCopyOnRead.read_57_disablecopyonread_adam_v_dense_42_bias"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp.read_57_disablecopyonread_adam_v_dense_42_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
:`�
Read_58/DisableCopyOnReadDisableCopyOnRead0read_58_disablecopyonread_adam_m_dense_47_kernel"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp0read_58_disablecopyonread_adam_m_dense_47_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0p
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`g
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_59/DisableCopyOnReadDisableCopyOnRead0read_59_disablecopyonread_adam_v_dense_47_kernel"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp0read_59_disablecopyonread_adam_v_dense_47_kernel^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0p
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`g
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_60/DisableCopyOnReadDisableCopyOnRead.read_60_disablecopyonread_adam_m_dense_47_bias"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp.read_60_disablecopyonread_adam_m_dense_47_bias^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_61/DisableCopyOnReadDisableCopyOnRead.read_61_disablecopyonread_adam_v_dense_47_bias"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp.read_61_disablecopyonread_adam_v_dense_47_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_62/DisableCopyOnReadDisableCopyOnRead0read_62_disablecopyonread_adam_m_dense_43_kernel"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp0read_62_disablecopyonread_adam_m_dense_43_kernel^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0p
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`g
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_63/DisableCopyOnReadDisableCopyOnRead0read_63_disablecopyonread_adam_v_dense_43_kernel"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp0read_63_disablecopyonread_adam_v_dense_43_kernel^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`*
dtype0p
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`g
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes

:`�
Read_64/DisableCopyOnReadDisableCopyOnRead.read_64_disablecopyonread_adam_m_dense_43_bias"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp.read_64_disablecopyonread_adam_m_dense_43_bias^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_65/DisableCopyOnReadDisableCopyOnRead.read_65_disablecopyonread_adam_v_dense_43_bias"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp.read_65_disablecopyonread_adam_v_dense_43_bias^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_66/DisableCopyOnReadDisableCopyOnRead!read_66_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp!read_66_disablecopyonread_total_2^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_67/DisableCopyOnReadDisableCopyOnRead!read_67_disablecopyonread_count_2"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp!read_67_disablecopyonread_count_2^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_68/DisableCopyOnReadDisableCopyOnRead!read_68_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp!read_68_disablecopyonread_total_1^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_69/DisableCopyOnReadDisableCopyOnRead!read_69_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp!read_69_disablecopyonread_count_1^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_70/DisableCopyOnReadDisableCopyOnReadread_70_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOpread_70_disablecopyonread_total^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_71/DisableCopyOnReadDisableCopyOnReadread_71_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOpread_71_disablecopyonread_count^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*�
value�B�IB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*�
value�B�IB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *W
dtypesM
K2I	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_144Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_145IdentityIdentity_144:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_145Identity_145:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=I9

_output_shapes
: 

_user_specified_nameConst:%H!

_user_specified_namecount:%G!

_user_specified_nametotal:'F#
!
_user_specified_name	count_1:'E#
!
_user_specified_name	total_1:'D#
!
_user_specified_name	count_2:'C#
!
_user_specified_name	total_2:4B0
.
_user_specified_nameAdam/v/dense_43/bias:4A0
.
_user_specified_nameAdam/m/dense_43/bias:6@2
0
_user_specified_nameAdam/v/dense_43/kernel:6?2
0
_user_specified_nameAdam/m/dense_43/kernel:4>0
.
_user_specified_nameAdam/v/dense_47/bias:4=0
.
_user_specified_nameAdam/m/dense_47/bias:6<2
0
_user_specified_nameAdam/v/dense_47/kernel:6;2
0
_user_specified_nameAdam/m/dense_47/kernel:4:0
.
_user_specified_nameAdam/v/dense_42/bias:490
.
_user_specified_nameAdam/m/dense_42/bias:682
0
_user_specified_nameAdam/v/dense_42/kernel:672
0
_user_specified_nameAdam/m/dense_42/kernel:460
.
_user_specified_nameAdam/v/dense_46/bias:450
.
_user_specified_nameAdam/m/dense_46/bias:642
0
_user_specified_nameAdam/v/dense_46/kernel:632
0
_user_specified_nameAdam/m/dense_46/kernel:B2>
<
_user_specified_name$"Adam/v/batch_normalization_10/beta:B1>
<
_user_specified_name$"Adam/m/batch_normalization_10/beta:C0?
=
_user_specified_name%#Adam/v/batch_normalization_10/gamma:C/?
=
_user_specified_name%#Adam/m/batch_normalization_10/gamma:B.>
<
_user_specified_name$"Adam/v/batch_normalization_11/beta:B->
<
_user_specified_name$"Adam/m/batch_normalization_11/beta:C,?
=
_user_specified_name%#Adam/v/batch_normalization_11/gamma:C+?
=
_user_specified_name%#Adam/m/batch_normalization_11/gamma:4*0
.
_user_specified_nameAdam/v/dense_41/bias:4)0
.
_user_specified_nameAdam/m/dense_41/bias:6(2
0
_user_specified_nameAdam/v/dense_41/kernel:6'2
0
_user_specified_nameAdam/m/dense_41/kernel:4&0
.
_user_specified_nameAdam/v/dense_45/bias:4%0
.
_user_specified_nameAdam/m/dense_45/bias:6$2
0
_user_specified_nameAdam/v/dense_45/kernel:6#2
0
_user_specified_nameAdam/m/dense_45/kernel:4"0
.
_user_specified_nameAdam/v/dense_40/bias:4!0
.
_user_specified_nameAdam/m/dense_40/bias:6 2
0
_user_specified_nameAdam/v/dense_40/kernel:62
0
_user_specified_nameAdam/m/dense_40/kernel:40
.
_user_specified_nameAdam/v/dense_44/bias:40
.
_user_specified_nameAdam/m/dense_44/bias:62
0
_user_specified_nameAdam/v/dense_44/kernel:62
0
_user_specified_nameAdam/m/dense_44/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_43/bias:/+
)
_user_specified_namedense_43/kernel:-)
'
_user_specified_namedense_47/bias:/+
)
_user_specified_namedense_47/kernel:-)
'
_user_specified_namedense_42/bias:/+
)
_user_specified_namedense_42/kernel:-)
'
_user_specified_namedense_46/bias:/+
)
_user_specified_namedense_46/kernel:FB
@
_user_specified_name(&batch_normalization_10/moving_variance:B>
<
_user_specified_name$"batch_normalization_10/moving_mean:;7
5
_user_specified_namebatch_normalization_10/beta:<8
6
_user_specified_namebatch_normalization_10/gamma:FB
@
_user_specified_name(&batch_normalization_11/moving_variance:B>
<
_user_specified_name$"batch_normalization_11/moving_mean:;
7
5
_user_specified_namebatch_normalization_11/beta:<	8
6
_user_specified_namebatch_normalization_11/gamma:-)
'
_user_specified_namedense_41/bias:/+
)
_user_specified_namedense_41/kernel:-)
'
_user_specified_namedense_45/bias:/+
)
_user_specified_namedense_45/kernel:-)
'
_user_specified_namedense_40/bias:/+
)
_user_specified_namedense_40/kernel:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
*__inference_dense_42_layer_call_fn_1805015

inputs
unknown:`
	unknown_0:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_42_layer_call_and_return_conditional_losses_1804075o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1805011:'#
!
_user_specified_name	1805009:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
L
$__inference__update_step_xla_1804586
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:`: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:`
"
_user_specified_name
gradient
�

f
G__inference_dropout_10_layer_call_and_return_conditional_losses_1805001

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������`Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������`*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������`T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������`a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�j
�
D__inference_model_5_layer_call_and_return_conditional_losses_1804132
pixel_projection_y

angles
cluster_charge#
dense_44_1803926:	�
dense_44_1803928:	�#
dense_40_1803947:	�
dense_40_1803949:	�#
dense_45_1803962:	�`
dense_45_1803964:`#
dense_41_1803989:	�`
dense_41_1803991:`,
batch_normalization_11_1803994:`,
batch_normalization_11_1803996:`,
batch_normalization_11_1803998:`,
batch_normalization_11_1804000:`"
dense_46_1804032:`
dense_46_1804034:`,
batch_normalization_10_1804037:`,
batch_normalization_10_1804039:`,
batch_normalization_10_1804041:`,
batch_normalization_10_1804043:`"
dense_42_1804076:`
dense_42_1804078:`"
dense_47_1804092:`
dense_47_1804094:"
dense_43_1804114:`
dense_43_1804116:
identity��.batch_normalization_10/StatefulPartitionedCall�.batch_normalization_11/StatefulPartitionedCall� dense_40/StatefulPartitionedCall� dense_41/StatefulPartitionedCall� dense_42/StatefulPartitionedCall� dense_43/StatefulPartitionedCall� dense_44/StatefulPartitionedCall� dense_45/StatefulPartitionedCall� dense_46/StatefulPartitionedCall� dense_47/StatefulPartitionedCall�"dropout_10/StatefulPartitionedCall�"dropout_11/StatefulPartitionedCall�
flatten_5/PartitionedCallPartitionedCallpixel_projection_y*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_1803905�
concatenate_10/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0anglescluster_charge*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_concatenate_10_layer_call_and_return_conditional_losses_1803914�
 dense_44/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0dense_44_1803926dense_44_1803928*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_44_layer_call_and_return_conditional_losses_1803925�
leaky_re_lu_22/PartitionedCallPartitionedCall)dense_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1803935�
 dense_40/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0dense_40_1803947dense_40_1803949*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_1803946�
 dense_45/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_22/PartitionedCall:output:0dense_45_1803962dense_45_1803964*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_45_layer_call_and_return_conditional_losses_1803961�
leaky_re_lu_20/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_1803971�
leaky_re_lu_23/PartitionedCallPartitionedCall)dense_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1803977�
 dense_41/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_20/PartitionedCall:output:0dense_41_1803989dense_41_1803991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_1803988�
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0batch_normalization_11_1803994batch_normalization_11_1803996batch_normalization_11_1803998batch_normalization_11_1804000*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1803769�
leaky_re_lu_21/PartitionedCallPartitionedCall)dense_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1804007�
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804020�
 dense_46/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0dense_46_1804032dense_46_1804034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_46_layer_call_and_return_conditional_losses_1804031�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0batch_normalization_10_1804037batch_normalization_10_1804039batch_normalization_10_1804041batch_normalization_10_1804043*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1803849�
add_11/PartitionedCallPartitionedCall+dropout_11/StatefulPartitionedCall:output:0)dense_46/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_add_11_layer_call_and_return_conditional_losses_1804051�
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_1804064�
 dense_42/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0dense_42_1804076dense_42_1804078*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_42_layer_call_and_return_conditional_losses_1804075�
 dense_47/StatefulPartitionedCallStatefulPartitionedCalladd_11/PartitionedCall:output:0dense_47_1804092dense_47_1804094*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_47_layer_call_and_return_conditional_losses_1804091�
add_10/PartitionedCallPartitionedCall+dropout_10/StatefulPartitionedCall:output:0)dense_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_add_10_layer_call_and_return_conditional_losses_1804102�
 dense_43/StatefulPartitionedCallStatefulPartitionedCalladd_10/PartitionedCall:output:0dense_43_1804114dense_43_1804116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_1804113o
*tf.clip_by_value_5/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
(tf.clip_by_value_5/clip_by_value/MinimumMinimum)dense_47/StatefulPartitionedCall:output:03tf.clip_by_value_5/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������g
"tf.clip_by_value_5/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@�
 tf.clip_by_value_5/clip_by_valueMaximum,tf.clip_by_value_5/clip_by_value/Minimum:z:0+tf.clip_by_value_5/clip_by_value/y:output:0*
T0*'
_output_shapes
:����������
concatenate_11/PartitionedCallPartitionedCall)dense_43/StatefulPartitionedCall:output:0$tf.clip_by_value_5/clip_by_value:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_concatenate_11_layer_call_and_return_conditional_losses_1804129v
IdentityIdentity'concatenate_11/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:���������:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall:'#
!
_user_specified_name	1804116:'#
!
_user_specified_name	1804114:'#
!
_user_specified_name	1804094:'#
!
_user_specified_name	1804092:'#
!
_user_specified_name	1804078:'#
!
_user_specified_name	1804076:'#
!
_user_specified_name	1804043:'#
!
_user_specified_name	1804041:'#
!
_user_specified_name	1804039:'#
!
_user_specified_name	1804037:'#
!
_user_specified_name	1804034:'#
!
_user_specified_name	1804032:'#
!
_user_specified_name	1804000:'#
!
_user_specified_name	1803998:'#
!
_user_specified_name	1803996:'#
!
_user_specified_name	1803994:'
#
!
_user_specified_name	1803991:'	#
!
_user_specified_name	1803989:'#
!
_user_specified_name	1803964:'#
!
_user_specified_name	1803962:'#
!
_user_specified_name	1803949:'#
!
_user_specified_name	1803947:'#
!
_user_specified_name	1803928:'#
!
_user_specified_name	1803926:WS
'
_output_shapes
:���������
(
_user_specified_namecluster_charge:OK
'
_output_shapes
:���������
 
_user_specified_nameangles:_ [
+
_output_shapes
:���������
,
_user_specified_namepixel_projection_y
�
g
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1803935

inputs
identityX
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:����������*
alpha%
�#<`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_flatten_5_layer_call_and_return_conditional_losses_1804642

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
\
0__inference_concatenate_11_layer_call_fn_1805094
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_concatenate_11_layer_call_and_return_conditional_losses_1804129`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
u
K__inference_concatenate_11_layer_call_and_return_conditional_losses_1804129

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_1803971

inputs
identityX
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:����������*
alpha%
�#<`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_dense_44_layer_call_and_return_conditional_losses_1803925

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_10_layer_call_fn_1804879

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1803869o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1804875:'#
!
_user_specified_name	1804873:'#
!
_user_specified_name	1804871:'#
!
_user_specified_name	1804869:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
L
$__inference__update_step_xla_1804571
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:`: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:`
"
_user_specified_name
gradient
�
L
0__inference_leaky_re_lu_23_layer_call_fn_1804758

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1803977`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�

f
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804955

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������`Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������`*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������`T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������`a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
M
$__inference__update_step_xla_1804541
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�	
�
E__inference_dense_44_layer_call_and_return_conditional_losses_1804676

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_43_layer_call_fn_1805078

inputs
unknown:`
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_1804113o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1805074:'#
!
_user_specified_name	1805072:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_11_layer_call_fn_1804799

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1803789o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1804795:'#
!
_user_specified_name	1804793:'#
!
_user_specified_name	1804791:'#
!
_user_specified_name	1804789:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
L
$__inference__update_step_xla_1804581
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:`: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:`
"
_user_specified_name
gradient
�
P
$__inference__update_step_xla_1804616
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:`: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:`
"
_user_specified_name
gradient
�
m
C__inference_add_10_layer_call_and_return_conditional_losses_1804102

inputs
inputs_1
identityP
addAddV2inputsinputs_1*
T0*'
_output_shapes
:���������`O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������`:���������`:OK
'
_output_shapes
:���������`
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�

�
E__inference_dense_47_layer_call_and_return_conditional_losses_1804091

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������X
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������e
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
K__inference_concatenate_10_layer_call_and_return_conditional_losses_1803914

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������:���������:���������:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1803769

inputs5
'assignmovingavg_readvariableop_resource:`7
)assignmovingavg_1_readvariableop_resource:`3
%batchnorm_mul_readvariableop_resource:`/
!batchnorm_readvariableop_resource:`
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:`*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:`�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������`l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:`*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:`*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:`x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:`�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:`*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:`~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:`�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:`P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:`~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������`h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:`v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:`r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������`b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
L
0__inference_leaky_re_lu_20_layer_call_fn_1804710

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_1803971a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_flatten_5_layer_call_and_return_conditional_losses_1803905

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_dense_41_layer_call_and_return_conditional_losses_1803988

inputs1
matmul_readvariableop_resource:	�`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������`S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
$__inference__update_step_xla_1804591
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:`: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:`
"
_user_specified_name
gradient
��
�-
#__inference__traced_restore_1805782
file_prefix3
 assignvariableop_dense_44_kernel:	�/
 assignvariableop_1_dense_44_bias:	�5
"assignvariableop_2_dense_40_kernel:	�/
 assignvariableop_3_dense_40_bias:	�5
"assignvariableop_4_dense_45_kernel:	�`.
 assignvariableop_5_dense_45_bias:`5
"assignvariableop_6_dense_41_kernel:	�`.
 assignvariableop_7_dense_41_bias:`=
/assignvariableop_8_batch_normalization_11_gamma:`<
.assignvariableop_9_batch_normalization_11_beta:`D
6assignvariableop_10_batch_normalization_11_moving_mean:`H
:assignvariableop_11_batch_normalization_11_moving_variance:`>
0assignvariableop_12_batch_normalization_10_gamma:`=
/assignvariableop_13_batch_normalization_10_beta:`D
6assignvariableop_14_batch_normalization_10_moving_mean:`H
:assignvariableop_15_batch_normalization_10_moving_variance:`5
#assignvariableop_16_dense_46_kernel:`/
!assignvariableop_17_dense_46_bias:`5
#assignvariableop_18_dense_42_kernel:`/
!assignvariableop_19_dense_42_bias:`5
#assignvariableop_20_dense_47_kernel:`/
!assignvariableop_21_dense_47_bias:5
#assignvariableop_22_dense_43_kernel:`/
!assignvariableop_23_dense_43_bias:'
assignvariableop_24_iteration:	 +
!assignvariableop_25_learning_rate: =
*assignvariableop_26_adam_m_dense_44_kernel:	�=
*assignvariableop_27_adam_v_dense_44_kernel:	�7
(assignvariableop_28_adam_m_dense_44_bias:	�7
(assignvariableop_29_adam_v_dense_44_bias:	�=
*assignvariableop_30_adam_m_dense_40_kernel:	�=
*assignvariableop_31_adam_v_dense_40_kernel:	�7
(assignvariableop_32_adam_m_dense_40_bias:	�7
(assignvariableop_33_adam_v_dense_40_bias:	�=
*assignvariableop_34_adam_m_dense_45_kernel:	�`=
*assignvariableop_35_adam_v_dense_45_kernel:	�`6
(assignvariableop_36_adam_m_dense_45_bias:`6
(assignvariableop_37_adam_v_dense_45_bias:`=
*assignvariableop_38_adam_m_dense_41_kernel:	�`=
*assignvariableop_39_adam_v_dense_41_kernel:	�`6
(assignvariableop_40_adam_m_dense_41_bias:`6
(assignvariableop_41_adam_v_dense_41_bias:`E
7assignvariableop_42_adam_m_batch_normalization_11_gamma:`E
7assignvariableop_43_adam_v_batch_normalization_11_gamma:`D
6assignvariableop_44_adam_m_batch_normalization_11_beta:`D
6assignvariableop_45_adam_v_batch_normalization_11_beta:`E
7assignvariableop_46_adam_m_batch_normalization_10_gamma:`E
7assignvariableop_47_adam_v_batch_normalization_10_gamma:`D
6assignvariableop_48_adam_m_batch_normalization_10_beta:`D
6assignvariableop_49_adam_v_batch_normalization_10_beta:`<
*assignvariableop_50_adam_m_dense_46_kernel:`<
*assignvariableop_51_adam_v_dense_46_kernel:`6
(assignvariableop_52_adam_m_dense_46_bias:`6
(assignvariableop_53_adam_v_dense_46_bias:`<
*assignvariableop_54_adam_m_dense_42_kernel:`<
*assignvariableop_55_adam_v_dense_42_kernel:`6
(assignvariableop_56_adam_m_dense_42_bias:`6
(assignvariableop_57_adam_v_dense_42_bias:`<
*assignvariableop_58_adam_m_dense_47_kernel:`<
*assignvariableop_59_adam_v_dense_47_kernel:`6
(assignvariableop_60_adam_m_dense_47_bias:6
(assignvariableop_61_adam_v_dense_47_bias:<
*assignvariableop_62_adam_m_dense_43_kernel:`<
*assignvariableop_63_adam_v_dense_43_kernel:`6
(assignvariableop_64_adam_m_dense_43_bias:6
(assignvariableop_65_adam_v_dense_43_bias:%
assignvariableop_66_total_2: %
assignvariableop_67_count_2: %
assignvariableop_68_total_1: %
assignvariableop_69_count_1: #
assignvariableop_70_total: #
assignvariableop_71_count: 
identity_73��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*�
value�B�IB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*�
value�B�IB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*W
dtypesM
K2I	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_44_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_44_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_40_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_40_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_45_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_45_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_41_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_41_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_11_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_11_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_11_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_11_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_10_gammaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_10_betaIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_10_moving_meanIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_10_moving_varianceIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_46_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_46_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_42_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_42_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_47_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_47_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_43_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_43_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_iterationIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_learning_rateIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_m_dense_44_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_v_dense_44_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_m_dense_44_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_v_dense_44_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_m_dense_40_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_v_dense_40_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_m_dense_40_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_v_dense_40_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_m_dense_45_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_v_dense_45_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_m_dense_45_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_v_dense_45_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_m_dense_41_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_v_dense_41_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_m_dense_41_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_v_dense_41_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp7assignvariableop_42_adam_m_batch_normalization_11_gammaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_v_batch_normalization_11_gammaIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_m_batch_normalization_11_betaIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_v_batch_normalization_11_betaIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp7assignvariableop_46_adam_m_batch_normalization_10_gammaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adam_v_batch_normalization_10_gammaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adam_m_batch_normalization_10_betaIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_v_batch_normalization_10_betaIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_m_dense_46_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_v_dense_46_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_m_dense_46_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_v_dense_46_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_m_dense_42_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_v_dense_42_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_m_dense_42_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_v_dense_42_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_m_dense_47_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_v_dense_47_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_m_dense_47_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_v_dense_47_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_m_dense_43_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_v_dense_43_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_m_dense_43_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_v_dense_43_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOpassignvariableop_66_total_2Identity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOpassignvariableop_67_count_2Identity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOpassignvariableop_68_total_1Identity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOpassignvariableop_69_count_1Identity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOpassignvariableop_70_totalIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOpassignvariableop_71_countIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_72Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_73IdentityIdentity_72:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_73Identity_73:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%H!

_user_specified_namecount:%G!

_user_specified_nametotal:'F#
!
_user_specified_name	count_1:'E#
!
_user_specified_name	total_1:'D#
!
_user_specified_name	count_2:'C#
!
_user_specified_name	total_2:4B0
.
_user_specified_nameAdam/v/dense_43/bias:4A0
.
_user_specified_nameAdam/m/dense_43/bias:6@2
0
_user_specified_nameAdam/v/dense_43/kernel:6?2
0
_user_specified_nameAdam/m/dense_43/kernel:4>0
.
_user_specified_nameAdam/v/dense_47/bias:4=0
.
_user_specified_nameAdam/m/dense_47/bias:6<2
0
_user_specified_nameAdam/v/dense_47/kernel:6;2
0
_user_specified_nameAdam/m/dense_47/kernel:4:0
.
_user_specified_nameAdam/v/dense_42/bias:490
.
_user_specified_nameAdam/m/dense_42/bias:682
0
_user_specified_nameAdam/v/dense_42/kernel:672
0
_user_specified_nameAdam/m/dense_42/kernel:460
.
_user_specified_nameAdam/v/dense_46/bias:450
.
_user_specified_nameAdam/m/dense_46/bias:642
0
_user_specified_nameAdam/v/dense_46/kernel:632
0
_user_specified_nameAdam/m/dense_46/kernel:B2>
<
_user_specified_name$"Adam/v/batch_normalization_10/beta:B1>
<
_user_specified_name$"Adam/m/batch_normalization_10/beta:C0?
=
_user_specified_name%#Adam/v/batch_normalization_10/gamma:C/?
=
_user_specified_name%#Adam/m/batch_normalization_10/gamma:B.>
<
_user_specified_name$"Adam/v/batch_normalization_11/beta:B->
<
_user_specified_name$"Adam/m/batch_normalization_11/beta:C,?
=
_user_specified_name%#Adam/v/batch_normalization_11/gamma:C+?
=
_user_specified_name%#Adam/m/batch_normalization_11/gamma:4*0
.
_user_specified_nameAdam/v/dense_41/bias:4)0
.
_user_specified_nameAdam/m/dense_41/bias:6(2
0
_user_specified_nameAdam/v/dense_41/kernel:6'2
0
_user_specified_nameAdam/m/dense_41/kernel:4&0
.
_user_specified_nameAdam/v/dense_45/bias:4%0
.
_user_specified_nameAdam/m/dense_45/bias:6$2
0
_user_specified_nameAdam/v/dense_45/kernel:6#2
0
_user_specified_nameAdam/m/dense_45/kernel:4"0
.
_user_specified_nameAdam/v/dense_40/bias:4!0
.
_user_specified_nameAdam/m/dense_40/bias:6 2
0
_user_specified_nameAdam/v/dense_40/kernel:62
0
_user_specified_nameAdam/m/dense_40/kernel:40
.
_user_specified_nameAdam/v/dense_44/bias:40
.
_user_specified_nameAdam/m/dense_44/bias:62
0
_user_specified_nameAdam/v/dense_44/kernel:62
0
_user_specified_nameAdam/m/dense_44/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_43/bias:/+
)
_user_specified_namedense_43/kernel:-)
'
_user_specified_namedense_47/bias:/+
)
_user_specified_namedense_47/kernel:-)
'
_user_specified_namedense_42/bias:/+
)
_user_specified_namedense_42/kernel:-)
'
_user_specified_namedense_46/bias:/+
)
_user_specified_namedense_46/kernel:FB
@
_user_specified_name(&batch_normalization_10/moving_variance:B>
<
_user_specified_name$"batch_normalization_10/moving_mean:;7
5
_user_specified_namebatch_normalization_10/beta:<8
6
_user_specified_namebatch_normalization_10/gamma:FB
@
_user_specified_name(&batch_normalization_11/moving_variance:B>
<
_user_specified_name$"batch_normalization_11/moving_mean:;
7
5
_user_specified_namebatch_normalization_11/beta:<	8
6
_user_specified_namebatch_normalization_11/gamma:-)
'
_user_specified_namedense_41/bias:/+
)
_user_specified_namedense_41/kernel:-)
'
_user_specified_namedense_45/bias:/+
)
_user_specified_namedense_45/kernel:-)
'
_user_specified_namedense_40/bias:/+
)
_user_specified_namedense_40/kernel:-)
'
_user_specified_namedense_44/bias:/+
)
_user_specified_namedense_44/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
P
$__inference__update_step_xla_1804606
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:`: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:`
"
_user_specified_name
gradient
�	
�
E__inference_dense_46_layer_call_and_return_conditional_losses_1804979

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������`S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
L
$__inference__update_step_xla_1804561
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:`: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:`
"
_user_specified_name
gradient
�
e
G__inference_dropout_10_layer_call_and_return_conditional_losses_1805006

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������`[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�	
�
E__inference_dense_41_layer_call_and_return_conditional_losses_1804753

inputs1
matmul_readvariableop_resource:	�`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������`S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804176

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������`[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
j
0__inference_concatenate_10_layer_call_fn_1804649
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_concatenate_10_layer_call_and_return_conditional_losses_1803914`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������:���������:���������:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
Q
$__inference__update_step_xla_1804556
gradient
variable:	�`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�`: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	�`
"
_user_specified_name
gradient
�
P
$__inference__update_step_xla_1804596
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:`: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:`
"
_user_specified_name
gradient
�
L
$__inference__update_step_xla_1804601
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:`: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:`
"
_user_specified_name
gradient
�	
�
E__inference_dense_40_layer_call_and_return_conditional_losses_1804695

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
Q
$__inference__update_step_xla_1804566
gradient
variable:	�`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�`: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	�`
"
_user_specified_name
gradient
�

f
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804020

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������`Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������`*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������`T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������`a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�	
�
E__inference_dense_45_layer_call_and_return_conditional_losses_1803961

inputs1
matmul_readvariableop_resource:	�`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������`S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�&
�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1803849

inputs5
'assignmovingavg_readvariableop_resource:`7
)assignmovingavg_1_readvariableop_resource:`3
%batchnorm_mul_readvariableop_resource:`/
!batchnorm_readvariableop_resource:`
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:`*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:`�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������`l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:`*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:`*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:`x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:`�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:`*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:`~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:`�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:`P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:`~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������`h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:`v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:`r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������`b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
*__inference_dense_45_layer_call_fn_1804724

inputs
unknown:	�`
	unknown_0:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_45_layer_call_and_return_conditional_losses_1803961o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1804720:'#
!
_user_specified_name	1804718:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_10_layer_call_fn_1804866

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1803849o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1804862:'#
!
_user_specified_name	1804860:'#
!
_user_specified_name	1804858:'#
!
_user_specified_name	1804856:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
e
G__inference_dropout_10_layer_call_and_return_conditional_losses_1804197

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������`[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
��
�
"__inference__wrapped_model_1803735
pixel_projection_y

angles
cluster_chargeB
/model_5_dense_44_matmul_readvariableop_resource:	�?
0model_5_dense_44_biasadd_readvariableop_resource:	�B
/model_5_dense_40_matmul_readvariableop_resource:	�?
0model_5_dense_40_biasadd_readvariableop_resource:	�B
/model_5_dense_45_matmul_readvariableop_resource:	�`>
0model_5_dense_45_biasadd_readvariableop_resource:`B
/model_5_dense_41_matmul_readvariableop_resource:	�`>
0model_5_dense_41_biasadd_readvariableop_resource:`N
@model_5_batch_normalization_11_batchnorm_readvariableop_resource:`R
Dmodel_5_batch_normalization_11_batchnorm_mul_readvariableop_resource:`P
Bmodel_5_batch_normalization_11_batchnorm_readvariableop_1_resource:`P
Bmodel_5_batch_normalization_11_batchnorm_readvariableop_2_resource:`A
/model_5_dense_46_matmul_readvariableop_resource:`>
0model_5_dense_46_biasadd_readvariableop_resource:`N
@model_5_batch_normalization_10_batchnorm_readvariableop_resource:`R
Dmodel_5_batch_normalization_10_batchnorm_mul_readvariableop_resource:`P
Bmodel_5_batch_normalization_10_batchnorm_readvariableop_1_resource:`P
Bmodel_5_batch_normalization_10_batchnorm_readvariableop_2_resource:`A
/model_5_dense_42_matmul_readvariableop_resource:`>
0model_5_dense_42_biasadd_readvariableop_resource:`A
/model_5_dense_47_matmul_readvariableop_resource:`>
0model_5_dense_47_biasadd_readvariableop_resource:A
/model_5_dense_43_matmul_readvariableop_resource:`>
0model_5_dense_43_biasadd_readvariableop_resource:
identity��7model_5/batch_normalization_10/batchnorm/ReadVariableOp�9model_5/batch_normalization_10/batchnorm/ReadVariableOp_1�9model_5/batch_normalization_10/batchnorm/ReadVariableOp_2�;model_5/batch_normalization_10/batchnorm/mul/ReadVariableOp�7model_5/batch_normalization_11/batchnorm/ReadVariableOp�9model_5/batch_normalization_11/batchnorm/ReadVariableOp_1�9model_5/batch_normalization_11/batchnorm/ReadVariableOp_2�;model_5/batch_normalization_11/batchnorm/mul/ReadVariableOp�'model_5/dense_40/BiasAdd/ReadVariableOp�&model_5/dense_40/MatMul/ReadVariableOp�'model_5/dense_41/BiasAdd/ReadVariableOp�&model_5/dense_41/MatMul/ReadVariableOp�'model_5/dense_42/BiasAdd/ReadVariableOp�&model_5/dense_42/MatMul/ReadVariableOp�'model_5/dense_43/BiasAdd/ReadVariableOp�&model_5/dense_43/MatMul/ReadVariableOp�'model_5/dense_44/BiasAdd/ReadVariableOp�&model_5/dense_44/MatMul/ReadVariableOp�'model_5/dense_45/BiasAdd/ReadVariableOp�&model_5/dense_45/MatMul/ReadVariableOp�'model_5/dense_46/BiasAdd/ReadVariableOp�&model_5/dense_46/MatMul/ReadVariableOp�'model_5/dense_47/BiasAdd/ReadVariableOp�&model_5/dense_47/MatMul/ReadVariableOph
model_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
model_5/flatten_5/ReshapeReshapepixel_projection_y model_5/flatten_5/Const:output:0*
T0*'
_output_shapes
:���������d
"model_5/concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_5/concatenate_10/concatConcatV2"model_5/flatten_5/Reshape:output:0anglescluster_charge+model_5/concatenate_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
&model_5/dense_44/MatMul/ReadVariableOpReadVariableOp/model_5_dense_44_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_5/dense_44/MatMulMatMul&model_5/concatenate_10/concat:output:0.model_5/dense_44/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'model_5/dense_44/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_44_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_5/dense_44/BiasAddBiasAdd!model_5/dense_44/MatMul:product:0/model_5/dense_44/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 model_5/leaky_re_lu_22/LeakyRelu	LeakyRelu!model_5/dense_44/BiasAdd:output:0*(
_output_shapes
:����������*
alpha%
�#<�
&model_5/dense_40/MatMul/ReadVariableOpReadVariableOp/model_5_dense_40_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_5/dense_40/MatMulMatMul&model_5/concatenate_10/concat:output:0.model_5/dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'model_5/dense_40/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_40_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_5/dense_40/BiasAddBiasAdd!model_5/dense_40/MatMul:product:0/model_5/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&model_5/dense_45/MatMul/ReadVariableOpReadVariableOp/model_5_dense_45_matmul_readvariableop_resource*
_output_shapes
:	�`*
dtype0�
model_5/dense_45/MatMulMatMul.model_5/leaky_re_lu_22/LeakyRelu:activations:0.model_5/dense_45/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
'model_5/dense_45/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_45_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0�
model_5/dense_45/BiasAddBiasAdd!model_5/dense_45/MatMul:product:0/model_5/dense_45/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
 model_5/leaky_re_lu_20/LeakyRelu	LeakyRelu!model_5/dense_40/BiasAdd:output:0*(
_output_shapes
:����������*
alpha%
�#<�
 model_5/leaky_re_lu_23/LeakyRelu	LeakyRelu!model_5/dense_45/BiasAdd:output:0*'
_output_shapes
:���������`*
alpha%
�#<�
&model_5/dense_41/MatMul/ReadVariableOpReadVariableOp/model_5_dense_41_matmul_readvariableop_resource*
_output_shapes
:	�`*
dtype0�
model_5/dense_41/MatMulMatMul.model_5/leaky_re_lu_20/LeakyRelu:activations:0.model_5/dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
'model_5/dense_41/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_41_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0�
model_5/dense_41/BiasAddBiasAdd!model_5/dense_41/MatMul:product:0/model_5/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
7model_5/batch_normalization_11/batchnorm/ReadVariableOpReadVariableOp@model_5_batch_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0s
.model_5/batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
,model_5/batch_normalization_11/batchnorm/addAddV2?model_5/batch_normalization_11/batchnorm/ReadVariableOp:value:07model_5/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
:`�
.model_5/batch_normalization_11/batchnorm/RsqrtRsqrt0model_5/batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
:`�
;model_5/batch_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_5_batch_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0�
,model_5/batch_normalization_11/batchnorm/mulMul2model_5/batch_normalization_11/batchnorm/Rsqrt:y:0Cmodel_5/batch_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`�
.model_5/batch_normalization_11/batchnorm/mul_1Mul.model_5/leaky_re_lu_23/LeakyRelu:activations:00model_5/batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������`�
9model_5/batch_normalization_11/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_5_batch_normalization_11_batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype0�
.model_5/batch_normalization_11/batchnorm/mul_2MulAmodel_5/batch_normalization_11/batchnorm/ReadVariableOp_1:value:00model_5/batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
:`�
9model_5/batch_normalization_11/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_5_batch_normalization_11_batchnorm_readvariableop_2_resource*
_output_shapes
:`*
dtype0�
,model_5/batch_normalization_11/batchnorm/subSubAmodel_5/batch_normalization_11/batchnorm/ReadVariableOp_2:value:02model_5/batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
:`�
.model_5/batch_normalization_11/batchnorm/add_1AddV22model_5/batch_normalization_11/batchnorm/mul_1:z:00model_5/batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������`�
 model_5/leaky_re_lu_21/LeakyRelu	LeakyRelu!model_5/dense_41/BiasAdd:output:0*'
_output_shapes
:���������`*
alpha%
�#<�
model_5/dropout_11/IdentityIdentity2model_5/batch_normalization_11/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������`�
&model_5/dense_46/MatMul/ReadVariableOpReadVariableOp/model_5_dense_46_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0�
model_5/dense_46/MatMulMatMul&model_5/concatenate_10/concat:output:0.model_5/dense_46/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
'model_5/dense_46/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_46_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0�
model_5/dense_46/BiasAddBiasAdd!model_5/dense_46/MatMul:product:0/model_5/dense_46/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
7model_5/batch_normalization_10/batchnorm/ReadVariableOpReadVariableOp@model_5_batch_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0s
.model_5/batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
,model_5/batch_normalization_10/batchnorm/addAddV2?model_5/batch_normalization_10/batchnorm/ReadVariableOp:value:07model_5/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:`�
.model_5/batch_normalization_10/batchnorm/RsqrtRsqrt0model_5/batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:`�
;model_5/batch_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_5_batch_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0�
,model_5/batch_normalization_10/batchnorm/mulMul2model_5/batch_normalization_10/batchnorm/Rsqrt:y:0Cmodel_5/batch_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`�
.model_5/batch_normalization_10/batchnorm/mul_1Mul.model_5/leaky_re_lu_21/LeakyRelu:activations:00model_5/batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������`�
9model_5/batch_normalization_10/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_5_batch_normalization_10_batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype0�
.model_5/batch_normalization_10/batchnorm/mul_2MulAmodel_5/batch_normalization_10/batchnorm/ReadVariableOp_1:value:00model_5/batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:`�
9model_5/batch_normalization_10/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_5_batch_normalization_10_batchnorm_readvariableop_2_resource*
_output_shapes
:`*
dtype0�
,model_5/batch_normalization_10/batchnorm/subSubAmodel_5/batch_normalization_10/batchnorm/ReadVariableOp_2:value:02model_5/batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:`�
.model_5/batch_normalization_10/batchnorm/add_1AddV22model_5/batch_normalization_10/batchnorm/mul_1:z:00model_5/batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������`�
model_5/add_11/addAddV2$model_5/dropout_11/Identity:output:0!model_5/dense_46/BiasAdd:output:0*
T0*'
_output_shapes
:���������`�
model_5/dropout_10/IdentityIdentity2model_5/batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������`�
&model_5/dense_42/MatMul/ReadVariableOpReadVariableOp/model_5_dense_42_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0�
model_5/dense_42/MatMulMatMul&model_5/concatenate_10/concat:output:0.model_5/dense_42/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
'model_5/dense_42/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_42_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0�
model_5/dense_42/BiasAddBiasAdd!model_5/dense_42/MatMul:product:0/model_5/dense_42/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`�
&model_5/dense_47/MatMul/ReadVariableOpReadVariableOp/model_5_dense_47_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0�
model_5/dense_47/MatMulMatMulmodel_5/add_11/add:z:0.model_5/dense_47/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_5/dense_47/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_47_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_5/dense_47/BiasAddBiasAdd!model_5/dense_47/MatMul:product:0/model_5/dense_47/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
model_5/dense_47/SoftplusSoftplus!model_5/dense_47/BiasAdd:output:0*
T0*'
_output_shapes
:����������
model_5/add_10/addAddV2$model_5/dropout_10/Identity:output:0!model_5/dense_42/BiasAdd:output:0*
T0*'
_output_shapes
:���������`�
&model_5/dense_43/MatMul/ReadVariableOpReadVariableOp/model_5_dense_43_matmul_readvariableop_resource*
_output_shapes

:`*
dtype0�
model_5/dense_43/MatMulMatMulmodel_5/add_10/add:z:0.model_5/dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_5/dense_43/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_43_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_5/dense_43/BiasAddBiasAdd!model_5/dense_43/MatMul:product:0/model_5/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
2model_5/tf.clip_by_value_5/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
0model_5/tf.clip_by_value_5/clip_by_value/MinimumMinimum'model_5/dense_47/Softplus:activations:0;model_5/tf.clip_by_value_5/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������o
*model_5/tf.clip_by_value_5/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@�
(model_5/tf.clip_by_value_5/clip_by_valueMaximum4model_5/tf.clip_by_value_5/clip_by_value/Minimum:z:03model_5/tf.clip_by_value_5/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������d
"model_5/concatenate_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_5/concatenate_11/concatConcatV2!model_5/dense_43/BiasAdd:output:0,model_5/tf.clip_by_value_5/clip_by_value:z:0+model_5/concatenate_11/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������u
IdentityIdentity&model_5/concatenate_11/concat:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp8^model_5/batch_normalization_10/batchnorm/ReadVariableOp:^model_5/batch_normalization_10/batchnorm/ReadVariableOp_1:^model_5/batch_normalization_10/batchnorm/ReadVariableOp_2<^model_5/batch_normalization_10/batchnorm/mul/ReadVariableOp8^model_5/batch_normalization_11/batchnorm/ReadVariableOp:^model_5/batch_normalization_11/batchnorm/ReadVariableOp_1:^model_5/batch_normalization_11/batchnorm/ReadVariableOp_2<^model_5/batch_normalization_11/batchnorm/mul/ReadVariableOp(^model_5/dense_40/BiasAdd/ReadVariableOp'^model_5/dense_40/MatMul/ReadVariableOp(^model_5/dense_41/BiasAdd/ReadVariableOp'^model_5/dense_41/MatMul/ReadVariableOp(^model_5/dense_42/BiasAdd/ReadVariableOp'^model_5/dense_42/MatMul/ReadVariableOp(^model_5/dense_43/BiasAdd/ReadVariableOp'^model_5/dense_43/MatMul/ReadVariableOp(^model_5/dense_44/BiasAdd/ReadVariableOp'^model_5/dense_44/MatMul/ReadVariableOp(^model_5/dense_45/BiasAdd/ReadVariableOp'^model_5/dense_45/MatMul/ReadVariableOp(^model_5/dense_46/BiasAdd/ReadVariableOp'^model_5/dense_46/MatMul/ReadVariableOp(^model_5/dense_47/BiasAdd/ReadVariableOp'^model_5/dense_47/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:���������:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : 2v
9model_5/batch_normalization_10/batchnorm/ReadVariableOp_19model_5/batch_normalization_10/batchnorm/ReadVariableOp_12v
9model_5/batch_normalization_10/batchnorm/ReadVariableOp_29model_5/batch_normalization_10/batchnorm/ReadVariableOp_22r
7model_5/batch_normalization_10/batchnorm/ReadVariableOp7model_5/batch_normalization_10/batchnorm/ReadVariableOp2z
;model_5/batch_normalization_10/batchnorm/mul/ReadVariableOp;model_5/batch_normalization_10/batchnorm/mul/ReadVariableOp2v
9model_5/batch_normalization_11/batchnorm/ReadVariableOp_19model_5/batch_normalization_11/batchnorm/ReadVariableOp_12v
9model_5/batch_normalization_11/batchnorm/ReadVariableOp_29model_5/batch_normalization_11/batchnorm/ReadVariableOp_22r
7model_5/batch_normalization_11/batchnorm/ReadVariableOp7model_5/batch_normalization_11/batchnorm/ReadVariableOp2z
;model_5/batch_normalization_11/batchnorm/mul/ReadVariableOp;model_5/batch_normalization_11/batchnorm/mul/ReadVariableOp2R
'model_5/dense_40/BiasAdd/ReadVariableOp'model_5/dense_40/BiasAdd/ReadVariableOp2P
&model_5/dense_40/MatMul/ReadVariableOp&model_5/dense_40/MatMul/ReadVariableOp2R
'model_5/dense_41/BiasAdd/ReadVariableOp'model_5/dense_41/BiasAdd/ReadVariableOp2P
&model_5/dense_41/MatMul/ReadVariableOp&model_5/dense_41/MatMul/ReadVariableOp2R
'model_5/dense_42/BiasAdd/ReadVariableOp'model_5/dense_42/BiasAdd/ReadVariableOp2P
&model_5/dense_42/MatMul/ReadVariableOp&model_5/dense_42/MatMul/ReadVariableOp2R
'model_5/dense_43/BiasAdd/ReadVariableOp'model_5/dense_43/BiasAdd/ReadVariableOp2P
&model_5/dense_43/MatMul/ReadVariableOp&model_5/dense_43/MatMul/ReadVariableOp2R
'model_5/dense_44/BiasAdd/ReadVariableOp'model_5/dense_44/BiasAdd/ReadVariableOp2P
&model_5/dense_44/MatMul/ReadVariableOp&model_5/dense_44/MatMul/ReadVariableOp2R
'model_5/dense_45/BiasAdd/ReadVariableOp'model_5/dense_45/BiasAdd/ReadVariableOp2P
&model_5/dense_45/MatMul/ReadVariableOp&model_5/dense_45/MatMul/ReadVariableOp2R
'model_5/dense_46/BiasAdd/ReadVariableOp'model_5/dense_46/BiasAdd/ReadVariableOp2P
&model_5/dense_46/MatMul/ReadVariableOp&model_5/dense_46/MatMul/ReadVariableOp2R
'model_5/dense_47/BiasAdd/ReadVariableOp'model_5/dense_47/BiasAdd/ReadVariableOp2P
&model_5/dense_47/MatMul/ReadVariableOp&model_5/dense_47/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:WS
'
_output_shapes
:���������
(
_user_specified_namecluster_charge:OK
'
_output_shapes
:���������
 
_user_specified_nameangles:_ [
+
_output_shapes
:���������
,
_user_specified_namepixel_projection_y
�
�
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1804853

inputs/
!batchnorm_readvariableop_resource:`3
%batchnorm_mul_readvariableop_resource:`1
#batchnorm_readvariableop_1_resource:`1
#batchnorm_readvariableop_2_resource:`
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:`P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:`~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������`z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:`z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:`*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:`r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������`b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1803789

inputs/
!batchnorm_readvariableop_resource:`3
%batchnorm_mul_readvariableop_resource:`1
#batchnorm_readvariableop_1_resource:`1
#batchnorm_readvariableop_2_resource:`
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:`P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:`~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������`z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:`z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:`*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:`r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������`b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_11_layer_call_fn_1804786

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1803769o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1804782:'#
!
_user_specified_name	1804780:'#
!
_user_specified_name	1804778:'#
!
_user_specified_name	1804776:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
e
,__inference_dropout_11_layer_call_fn_1804938

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804020o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
L
$__inference__update_step_xla_1804621
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
�g
�
D__inference_model_5_layer_call_and_return_conditional_losses_1804221
pixel_projection_y

angles
cluster_charge#
dense_44_1804139:	�
dense_44_1804141:	�#
dense_40_1804145:	�
dense_40_1804147:	�#
dense_45_1804150:	�`
dense_45_1804152:`#
dense_41_1804157:	�`
dense_41_1804159:`,
batch_normalization_11_1804162:`,
batch_normalization_11_1804164:`,
batch_normalization_11_1804166:`,
batch_normalization_11_1804168:`"
dense_46_1804178:`
dense_46_1804180:`,
batch_normalization_10_1804183:`,
batch_normalization_10_1804185:`,
batch_normalization_10_1804187:`,
batch_normalization_10_1804189:`"
dense_42_1804199:`
dense_42_1804201:`"
dense_47_1804204:`
dense_47_1804206:"
dense_43_1804210:`
dense_43_1804212:
identity��.batch_normalization_10/StatefulPartitionedCall�.batch_normalization_11/StatefulPartitionedCall� dense_40/StatefulPartitionedCall� dense_41/StatefulPartitionedCall� dense_42/StatefulPartitionedCall� dense_43/StatefulPartitionedCall� dense_44/StatefulPartitionedCall� dense_45/StatefulPartitionedCall� dense_46/StatefulPartitionedCall� dense_47/StatefulPartitionedCall�
flatten_5/PartitionedCallPartitionedCallpixel_projection_y*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_1803905�
concatenate_10/PartitionedCallPartitionedCall"flatten_5/PartitionedCall:output:0anglescluster_charge*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_concatenate_10_layer_call_and_return_conditional_losses_1803914�
 dense_44/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0dense_44_1804139dense_44_1804141*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_44_layer_call_and_return_conditional_losses_1803925�
leaky_re_lu_22/PartitionedCallPartitionedCall)dense_44/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1803935�
 dense_40/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0dense_40_1804145dense_40_1804147*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_1803946�
 dense_45/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_22/PartitionedCall:output:0dense_45_1804150dense_45_1804152*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_45_layer_call_and_return_conditional_losses_1803961�
leaky_re_lu_20/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_1803971�
leaky_re_lu_23/PartitionedCallPartitionedCall)dense_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1803977�
 dense_41/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_20/PartitionedCall:output:0dense_41_1804157dense_41_1804159*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_1803988�
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0batch_normalization_11_1804162batch_normalization_11_1804164batch_normalization_11_1804166batch_normalization_11_1804168*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1803789�
leaky_re_lu_21/PartitionedCallPartitionedCall)dense_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1804007�
dropout_11/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804176�
 dense_46/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0dense_46_1804178dense_46_1804180*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_46_layer_call_and_return_conditional_losses_1804031�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0batch_normalization_10_1804183batch_normalization_10_1804185batch_normalization_10_1804187batch_normalization_10_1804189*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1803869�
add_11/PartitionedCallPartitionedCall#dropout_11/PartitionedCall:output:0)dense_46/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_add_11_layer_call_and_return_conditional_losses_1804051�
dropout_10/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_1804197�
 dense_42/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0dense_42_1804199dense_42_1804201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_42_layer_call_and_return_conditional_losses_1804075�
 dense_47/StatefulPartitionedCallStatefulPartitionedCalladd_11/PartitionedCall:output:0dense_47_1804204dense_47_1804206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_47_layer_call_and_return_conditional_losses_1804091�
add_10/PartitionedCallPartitionedCall#dropout_10/PartitionedCall:output:0)dense_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *L
fGRE
C__inference_add_10_layer_call_and_return_conditional_losses_1804102�
 dense_43/StatefulPartitionedCallStatefulPartitionedCalladd_10/PartitionedCall:output:0dense_43_1804210dense_43_1804212*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_1804113o
*tf.clip_by_value_5/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
(tf.clip_by_value_5/clip_by_value/MinimumMinimum)dense_47/StatefulPartitionedCall:output:03tf.clip_by_value_5/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������g
"tf.clip_by_value_5/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@�
 tf.clip_by_value_5/clip_by_valueMaximum,tf.clip_by_value_5/clip_by_value/Minimum:z:0+tf.clip_by_value_5/clip_by_value/y:output:0*
T0*'
_output_shapes
:����������
concatenate_11/PartitionedCallPartitionedCall)dense_43/StatefulPartitionedCall:output:0$tf.clip_by_value_5/clip_by_value:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_concatenate_11_layer_call_and_return_conditional_losses_1804129v
IdentityIdentity'concatenate_11/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall!^dense_45/StatefulPartitionedCall!^dense_46/StatefulPartitionedCall!^dense_47/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:���������:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2D
 dense_45/StatefulPartitionedCall dense_45/StatefulPartitionedCall2D
 dense_46/StatefulPartitionedCall dense_46/StatefulPartitionedCall2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall:'#
!
_user_specified_name	1804212:'#
!
_user_specified_name	1804210:'#
!
_user_specified_name	1804206:'#
!
_user_specified_name	1804204:'#
!
_user_specified_name	1804201:'#
!
_user_specified_name	1804199:'#
!
_user_specified_name	1804189:'#
!
_user_specified_name	1804187:'#
!
_user_specified_name	1804185:'#
!
_user_specified_name	1804183:'#
!
_user_specified_name	1804180:'#
!
_user_specified_name	1804178:'#
!
_user_specified_name	1804168:'#
!
_user_specified_name	1804166:'#
!
_user_specified_name	1804164:'#
!
_user_specified_name	1804162:'
#
!
_user_specified_name	1804159:'	#
!
_user_specified_name	1804157:'#
!
_user_specified_name	1804152:'#
!
_user_specified_name	1804150:'#
!
_user_specified_name	1804147:'#
!
_user_specified_name	1804145:'#
!
_user_specified_name	1804141:'#
!
_user_specified_name	1804139:WS
'
_output_shapes
:���������
(
_user_specified_namecluster_charge:OK
'
_output_shapes
:���������
 
_user_specified_nameangles:_ [
+
_output_shapes
:���������
,
_user_specified_namepixel_projection_y
�
�
*__inference_dense_46_layer_call_fn_1804969

inputs
unknown:`
	unknown_0:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_46_layer_call_and_return_conditional_losses_1804031o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1804965:'#
!
_user_specified_name	1804963:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1804007

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������`*
alpha%
�#<_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�	
�
E__inference_dense_42_layer_call_and_return_conditional_losses_1804075

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������`S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_47_layer_call_and_return_conditional_losses_1805069

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������X
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������e
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
)__inference_model_5_layer_call_fn_1804276
pixel_projection_y

angles
cluster_charge
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�`
	unknown_4:`
	unknown_5:	�`
	unknown_6:`
	unknown_7:`
	unknown_8:`
	unknown_9:`

unknown_10:`

unknown_11:`

unknown_12:`

unknown_13:`

unknown_14:`

unknown_15:`

unknown_16:`

unknown_17:`

unknown_18:`

unknown_19:`

unknown_20:

unknown_21:`

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallpixel_projection_yanglescluster_chargeunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_model_5_layer_call_and_return_conditional_losses_1804132o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapeso
m:���������:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1804272:'#
!
_user_specified_name	1804270:'#
!
_user_specified_name	1804268:'#
!
_user_specified_name	1804266:'#
!
_user_specified_name	1804264:'#
!
_user_specified_name	1804262:'#
!
_user_specified_name	1804260:'#
!
_user_specified_name	1804258:'#
!
_user_specified_name	1804256:'#
!
_user_specified_name	1804254:'#
!
_user_specified_name	1804252:'#
!
_user_specified_name	1804250:'#
!
_user_specified_name	1804248:'#
!
_user_specified_name	1804246:'#
!
_user_specified_name	1804244:'#
!
_user_specified_name	1804242:'
#
!
_user_specified_name	1804240:'	#
!
_user_specified_name	1804238:'#
!
_user_specified_name	1804236:'#
!
_user_specified_name	1804234:'#
!
_user_specified_name	1804232:'#
!
_user_specified_name	1804230:'#
!
_user_specified_name	1804228:'#
!
_user_specified_name	1804226:WS
'
_output_shapes
:���������
(
_user_specified_namecluster_charge:OK
'
_output_shapes
:���������
 
_user_specified_nameangles:_ [
+
_output_shapes
:���������
,
_user_specified_namepixel_projection_y
�
L
$__inference__update_step_xla_1804611
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:`: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:`
"
_user_specified_name
gradient
�
g
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1804705

inputs
identityX
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:����������*
alpha%
�#<`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
P
$__inference__update_step_xla_1804626
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:`: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:`
"
_user_specified_name
gradient
�
m
C__inference_add_11_layer_call_and_return_conditional_losses_1804051

inputs
inputs_1
identityP
addAddV2inputsinputs_1*
T0*'
_output_shapes
:���������`O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������`:���������`:OK
'
_output_shapes
:���������`
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
o
C__inference_add_10_layer_call_and_return_conditional_losses_1805049
inputs_0
inputs_1
identityR
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:���������`O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������`:���������`:QM
'
_output_shapes
:���������`
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������`
"
_user_specified_name
inputs_0
�
g
K__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_1804715

inputs
identityX
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:����������*
alpha%
�#<`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_dropout_11_layer_call_fn_1804943

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804176`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1804933

inputs/
!batchnorm_readvariableop_resource:`3
%batchnorm_mul_readvariableop_resource:`1
#batchnorm_readvariableop_1_resource:`1
#batchnorm_readvariableop_2_resource:`
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:`*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:`P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:`~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:`*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:`c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������`z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:`*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:`z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:`*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:`r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������`b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������`�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������`: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�	
�
E__inference_dense_45_layer_call_and_return_conditional_losses_1804734

inputs1
matmul_readvariableop_resource:	�`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������`S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
K__inference_concatenate_10_layer_call_and_return_conditional_losses_1804657
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������:���������:���������:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
g
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1803977

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������`*
alpha%
�#<_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������`:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
Q
$__inference__update_step_xla_1804546
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	�
"
_user_specified_name
gradient
�
L
0__inference_leaky_re_lu_22_layer_call_fn_1804700

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1803935a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_41_layer_call_fn_1804743

inputs
unknown:	�`
	unknown_0:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_1803988o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1804739:'#
!
_user_specified_name	1804737:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
+__inference_flatten_5_layer_call_fn_1804636

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_1803905`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
angles/
serving_default_angles:0���������
I
cluster_charge7
 serving_default_cluster_charge:0���������
U
pixel_projection_y?
$serving_default_pixel_projection_y:0���������B
concatenate_110
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
layer-15
layer_with_weights-6
layer-16
layer-17
layer_with_weights-7
layer-18
layer-19
layer-20
layer_with_weights-8
layer-21
layer_with_weights-9
layer-22
layer-23
layer-24
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _default_save_signature
!	optimizer
"
signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance"
_tf_keras_layer
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
xaxis
	ygamma
zbeta
{moving_mean
|moving_variance"
_tf_keras_layer
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
50
61
=2
>3
Q4
R5
Y6
Z7
n8
o9
p10
q11
y12
z13
{14
|15
�16
�17
�18
�19
�20
�21
�22
�23"
trackable_list_wrapper
�
50
61
=2
>3
Q4
R5
Y6
Z7
n8
o9
y10
z11
�12
�13
�14
�15
�16
�17
�18
�19"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
 _default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
)__inference_model_5_layer_call_fn_1804276
)__inference_model_5_layer_call_fn_1804331�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
D__inference_model_5_layer_call_and_return_conditional_losses_1804132
D__inference_model_5_layer_call_and_return_conditional_losses_1804221�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
"__inference__wrapped_model_1803735pixel_projection_yanglescluster_charge"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_flatten_5_layer_call_fn_1804636�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_flatten_5_layer_call_and_return_conditional_losses_1804642�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_concatenate_10_layer_call_fn_1804649�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_concatenate_10_layer_call_and_return_conditional_losses_1804657�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_44_layer_call_fn_1804666�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_44_layer_call_and_return_conditional_losses_1804676�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 	�2dense_44/kernel
:�2dense_44/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_40_layer_call_fn_1804685�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_40_layer_call_and_return_conditional_losses_1804695�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 	�2dense_40/kernel
:�2dense_40/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_leaky_re_lu_22_layer_call_fn_1804700�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1804705�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_leaky_re_lu_20_layer_call_fn_1804710�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_1804715�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_45_layer_call_fn_1804724�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_45_layer_call_and_return_conditional_losses_1804734�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 	�`2dense_45/kernel
:`2dense_45/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_41_layer_call_fn_1804743�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_41_layer_call_and_return_conditional_losses_1804753�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 	�`2dense_41/kernel
:`2dense_41/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_leaky_re_lu_23_layer_call_fn_1804758�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1804763�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_leaky_re_lu_21_layer_call_fn_1804768�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1804773�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
<
n0
o1
p2
q3"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_11_layer_call_fn_1804786
8__inference_batch_normalization_11_layer_call_fn_1804799�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1804833
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1804853�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(`2batch_normalization_11/gamma
):'`2batch_normalization_11/beta
2:0` (2"batch_normalization_11/moving_mean
6:4` (2&batch_normalization_11/moving_variance
<
y0
z1
{2
|3"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
8__inference_batch_normalization_10_layer_call_fn_1804866
8__inference_batch_normalization_10_layer_call_fn_1804879�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1804913
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1804933�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(`2batch_normalization_10/gamma
):'`2batch_normalization_10/beta
2:0` (2"batch_normalization_10/moving_mean
6:4` (2&batch_normalization_10/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_11_layer_call_fn_1804938
,__inference_dropout_11_layer_call_fn_1804943�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804955
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804960�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_46_layer_call_fn_1804969�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_46_layer_call_and_return_conditional_losses_1804979�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:`2dense_46/kernel
:`2dense_46/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_10_layer_call_fn_1804984
,__inference_dropout_10_layer_call_fn_1804989�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_10_layer_call_and_return_conditional_losses_1805001
G__inference_dropout_10_layer_call_and_return_conditional_losses_1805006�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_42_layer_call_fn_1805015�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_42_layer_call_and_return_conditional_losses_1805025�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:`2dense_42/kernel
:`2dense_42/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_add_11_layer_call_fn_1805031�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_add_11_layer_call_and_return_conditional_losses_1805037�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_add_10_layer_call_fn_1805043�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_add_10_layer_call_and_return_conditional_losses_1805049�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_47_layer_call_fn_1805058�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_47_layer_call_and_return_conditional_losses_1805069�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:`2dense_47/kernel
:2dense_47/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_43_layer_call_fn_1805078�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_43_layer_call_and_return_conditional_losses_1805088�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:`2dense_43/kernel
:2dense_43/bias
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_concatenate_11_layer_call_fn_1805094�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_concatenate_11_layer_call_and_return_conditional_losses_1805101�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
<
p0
q1
{2
|3"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_model_5_layer_call_fn_1804276pixel_projection_yanglescluster_charge"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_5_layer_call_fn_1804331pixel_projection_yanglescluster_charge"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_5_layer_call_and_return_conditional_losses_1804132pixel_projection_yanglescluster_charge"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_5_layer_call_and_return_conditional_losses_1804221pixel_projection_yanglescluster_charge"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19"
trackable_list_wrapper
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11
�trace_12
�trace_13
�trace_14
�trace_15
�trace_16
�trace_17
�trace_18
�trace_192�
$__inference__update_step_xla_1804536
$__inference__update_step_xla_1804541
$__inference__update_step_xla_1804546
$__inference__update_step_xla_1804551
$__inference__update_step_xla_1804556
$__inference__update_step_xla_1804561
$__inference__update_step_xla_1804566
$__inference__update_step_xla_1804571
$__inference__update_step_xla_1804576
$__inference__update_step_xla_1804581
$__inference__update_step_xla_1804586
$__inference__update_step_xla_1804591
$__inference__update_step_xla_1804596
$__inference__update_step_xla_1804601
$__inference__update_step_xla_1804606
$__inference__update_step_xla_1804611
$__inference__update_step_xla_1804616
$__inference__update_step_xla_1804621
$__inference__update_step_xla_1804626
$__inference__update_step_xla_1804631�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7z�trace_8z�trace_9z�trace_10z�trace_11z�trace_12z�trace_13z�trace_14z�trace_15z�trace_16z�trace_17z�trace_18z�trace_19
�B�
%__inference_signature_wrapper_1804531anglescluster_chargepixel_projection_y"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_flatten_5_layer_call_fn_1804636inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_flatten_5_layer_call_and_return_conditional_losses_1804642inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
0__inference_concatenate_10_layer_call_fn_1804649inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_concatenate_10_layer_call_and_return_conditional_losses_1804657inputs_0inputs_1inputs_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_44_layer_call_fn_1804666inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_44_layer_call_and_return_conditional_losses_1804676inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_40_layer_call_fn_1804685inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_40_layer_call_and_return_conditional_losses_1804695inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
0__inference_leaky_re_lu_22_layer_call_fn_1804700inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1804705inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
0__inference_leaky_re_lu_20_layer_call_fn_1804710inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_1804715inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_45_layer_call_fn_1804724inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_45_layer_call_and_return_conditional_losses_1804734inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_41_layer_call_fn_1804743inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_41_layer_call_and_return_conditional_losses_1804753inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
0__inference_leaky_re_lu_23_layer_call_fn_1804758inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1804763inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
0__inference_leaky_re_lu_21_layer_call_fn_1804768inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1804773inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_11_layer_call_fn_1804786inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_batch_normalization_11_layer_call_fn_1804799inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1804833inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1804853inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
8__inference_batch_normalization_10_layer_call_fn_1804866inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
8__inference_batch_normalization_10_layer_call_fn_1804879inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1804913inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1804933inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
,__inference_dropout_11_layer_call_fn_1804938inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_11_layer_call_fn_1804943inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804955inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804960inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_46_layer_call_fn_1804969inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_46_layer_call_and_return_conditional_losses_1804979inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
,__inference_dropout_10_layer_call_fn_1804984inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_10_layer_call_fn_1804989inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_10_layer_call_and_return_conditional_losses_1805001inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_10_layer_call_and_return_conditional_losses_1805006inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_42_layer_call_fn_1805015inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_42_layer_call_and_return_conditional_losses_1805025inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_add_11_layer_call_fn_1805031inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_add_11_layer_call_and_return_conditional_losses_1805037inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_add_10_layer_call_fn_1805043inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_add_10_layer_call_and_return_conditional_losses_1805049inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_47_layer_call_fn_1805058inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_47_layer_call_and_return_conditional_losses_1805069inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_43_layer_call_fn_1805078inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_43_layer_call_and_return_conditional_losses_1805088inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
0__inference_concatenate_11_layer_call_fn_1805094inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_concatenate_11_layer_call_and_return_conditional_losses_1805101inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
':%	�2Adam/m/dense_44/kernel
':%	�2Adam/v/dense_44/kernel
!:�2Adam/m/dense_44/bias
!:�2Adam/v/dense_44/bias
':%	�2Adam/m/dense_40/kernel
':%	�2Adam/v/dense_40/kernel
!:�2Adam/m/dense_40/bias
!:�2Adam/v/dense_40/bias
':%	�`2Adam/m/dense_45/kernel
':%	�`2Adam/v/dense_45/kernel
 :`2Adam/m/dense_45/bias
 :`2Adam/v/dense_45/bias
':%	�`2Adam/m/dense_41/kernel
':%	�`2Adam/v/dense_41/kernel
 :`2Adam/m/dense_41/bias
 :`2Adam/v/dense_41/bias
/:-`2#Adam/m/batch_normalization_11/gamma
/:-`2#Adam/v/batch_normalization_11/gamma
.:,`2"Adam/m/batch_normalization_11/beta
.:,`2"Adam/v/batch_normalization_11/beta
/:-`2#Adam/m/batch_normalization_10/gamma
/:-`2#Adam/v/batch_normalization_10/gamma
.:,`2"Adam/m/batch_normalization_10/beta
.:,`2"Adam/v/batch_normalization_10/beta
&:$`2Adam/m/dense_46/kernel
&:$`2Adam/v/dense_46/kernel
 :`2Adam/m/dense_46/bias
 :`2Adam/v/dense_46/bias
&:$`2Adam/m/dense_42/kernel
&:$`2Adam/v/dense_42/kernel
 :`2Adam/m/dense_42/bias
 :`2Adam/v/dense_42/bias
&:$`2Adam/m/dense_47/kernel
&:$`2Adam/v/dense_47/kernel
 :2Adam/m/dense_47/bias
 :2Adam/v/dense_47/bias
&:$`2Adam/m/dense_43/kernel
&:$`2Adam/v/dense_43/kernel
 :2Adam/m/dense_43/bias
 :2Adam/v/dense_43/bias
�B�
$__inference__update_step_xla_1804536gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804541gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804546gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804551gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804556gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804561gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804566gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804571gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804576gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804581gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804586gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804591gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804596gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804601gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804606gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804611gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804616gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804621gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804626gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference__update_step_xla_1804631gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
$__inference__update_step_xla_1804536pj�g
`�]
�
gradient	�
5�2	�
�	�
�
p
` VariableSpec 
`��菧�?
� "
 �
$__inference__update_step_xla_1804541hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`��菧�?
� "
 �
$__inference__update_step_xla_1804546pj�g
`�]
�
gradient	�
5�2	�
�	�
�
p
` VariableSpec 
`��菧�?
� "
 �
$__inference__update_step_xla_1804551hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`��菧�?
� "
 �
$__inference__update_step_xla_1804556pj�g
`�]
�
gradient	�`
5�2	�
�	�`
�
p
` VariableSpec 
`��Ÿ�?
� "
 �
$__inference__update_step_xla_1804561f`�]
V�S
�
gradient`
0�-	�
�`
�
p
` VariableSpec 
`��Ÿ�?
� "
 �
$__inference__update_step_xla_1804566pj�g
`�]
�
gradient	�`
5�2	�
�	�`
�
p
` VariableSpec 
`�ۊ���?
� "
 �
$__inference__update_step_xla_1804571f`�]
V�S
�
gradient`
0�-	�
�`
�
p
` VariableSpec 
`������?
� "
 �
$__inference__update_step_xla_1804576f`�]
V�S
�
gradient`
0�-	�
�`
�
p
` VariableSpec 
`����?
� "
 �
$__inference__update_step_xla_1804581f`�]
V�S
�
gradient`
0�-	�
�`
�
p
` VariableSpec 
`����?
� "
 �
$__inference__update_step_xla_1804586f`�]
V�S
�
gradient`
0�-	�
�`
�
p
` VariableSpec 
`������?
� "
 �
$__inference__update_step_xla_1804591f`�]
V�S
�
gradient`
0�-	�
�`
�
p
` VariableSpec 
`������?
� "
 �
$__inference__update_step_xla_1804596nh�e
^�[
�
gradient`
4�1	�
�`
�
p
` VariableSpec 
`������?
� "
 �
$__inference__update_step_xla_1804601f`�]
V�S
�
gradient`
0�-	�
�`
�
p
` VariableSpec 
`��Ƕ�?
� "
 �
$__inference__update_step_xla_1804606nh�e
^�[
�
gradient`
4�1	�
�`
�
p
` VariableSpec 
`���ƶ�?
� "
 �
$__inference__update_step_xla_1804611f`�]
V�S
�
gradient`
0�-	�
�`
�
p
` VariableSpec 
`���ƶ�?
� "
 �
$__inference__update_step_xla_1804616nh�e
^�[
�
gradient`
4�1	�
�`
�
p
` VariableSpec 
`���Ÿ�?
� "
 �
$__inference__update_step_xla_1804621f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`༤Ÿ�?
� "
 �
$__inference__update_step_xla_1804626nh�e
^�[
�
gradient`
4�1	�
�`
�
p
` VariableSpec 
`��ǌ��?
� "
 �
$__inference__update_step_xla_1804631f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`�݈���?
� "
 �
"__inference__wrapped_model_1803735� 56=>QRYZqnpo��|y{z���������
���
��~
0�-
pixel_projection_y���������
 �
angles���������
(�%
cluster_charge���������
� "?�<
:
concatenate_11(�%
concatenate_11����������
C__inference_add_10_layer_call_and_return_conditional_losses_1805049�Z�W
P�M
K�H
"�
inputs_0���������`
"�
inputs_1���������`
� ",�)
"�
tensor_0���������`
� �
(__inference_add_10_layer_call_fn_1805043Z�W
P�M
K�H
"�
inputs_0���������`
"�
inputs_1���������`
� "!�
unknown���������`�
C__inference_add_11_layer_call_and_return_conditional_losses_1805037�Z�W
P�M
K�H
"�
inputs_0���������`
"�
inputs_1���������`
� ",�)
"�
tensor_0���������`
� �
(__inference_add_11_layer_call_fn_1805031Z�W
P�M
K�H
"�
inputs_0���������`
"�
inputs_1���������`
� "!�
unknown���������`�
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1804913m{|yz7�4
-�*
 �
inputs���������`
p

 
� ",�)
"�
tensor_0���������`
� �
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_1804933m|y{z7�4
-�*
 �
inputs���������`
p 

 
� ",�)
"�
tensor_0���������`
� �
8__inference_batch_normalization_10_layer_call_fn_1804866b{|yz7�4
-�*
 �
inputs���������`
p

 
� "!�
unknown���������`�
8__inference_batch_normalization_10_layer_call_fn_1804879b|y{z7�4
-�*
 �
inputs���������`
p 

 
� "!�
unknown���������`�
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1804833mpqno7�4
-�*
 �
inputs���������`
p

 
� ",�)
"�
tensor_0���������`
� �
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_1804853mqnpo7�4
-�*
 �
inputs���������`
p 

 
� ",�)
"�
tensor_0���������`
� �
8__inference_batch_normalization_11_layer_call_fn_1804786bpqno7�4
-�*
 �
inputs���������`
p

 
� "!�
unknown���������`�
8__inference_batch_normalization_11_layer_call_fn_1804799bqnpo7�4
-�*
 �
inputs���������`
p 

 
� "!�
unknown���������`�
K__inference_concatenate_10_layer_call_and_return_conditional_losses_1804657�~�{
t�q
o�l
"�
inputs_0���������
"�
inputs_1���������
"�
inputs_2���������
� ",�)
"�
tensor_0���������
� �
0__inference_concatenate_10_layer_call_fn_1804649�~�{
t�q
o�l
"�
inputs_0���������
"�
inputs_1���������
"�
inputs_2���������
� "!�
unknown����������
K__inference_concatenate_11_layer_call_and_return_conditional_losses_1805101�Z�W
P�M
K�H
"�
inputs_0���������
"�
inputs_1���������
� ",�)
"�
tensor_0���������
� �
0__inference_concatenate_11_layer_call_fn_1805094Z�W
P�M
K�H
"�
inputs_0���������
"�
inputs_1���������
� "!�
unknown����������
E__inference_dense_40_layer_call_and_return_conditional_losses_1804695d=>/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_40_layer_call_fn_1804685Y=>/�,
%�"
 �
inputs���������
� ""�
unknown�����������
E__inference_dense_41_layer_call_and_return_conditional_losses_1804753dYZ0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������`
� �
*__inference_dense_41_layer_call_fn_1804743YYZ0�-
&�#
!�
inputs����������
� "!�
unknown���������`�
E__inference_dense_42_layer_call_and_return_conditional_losses_1805025e��/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������`
� �
*__inference_dense_42_layer_call_fn_1805015Z��/�,
%�"
 �
inputs���������
� "!�
unknown���������`�
E__inference_dense_43_layer_call_and_return_conditional_losses_1805088e��/�,
%�"
 �
inputs���������`
� ",�)
"�
tensor_0���������
� �
*__inference_dense_43_layer_call_fn_1805078Z��/�,
%�"
 �
inputs���������`
� "!�
unknown����������
E__inference_dense_44_layer_call_and_return_conditional_losses_1804676d56/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_44_layer_call_fn_1804666Y56/�,
%�"
 �
inputs���������
� ""�
unknown�����������
E__inference_dense_45_layer_call_and_return_conditional_losses_1804734dQR0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������`
� �
*__inference_dense_45_layer_call_fn_1804724YQR0�-
&�#
!�
inputs����������
� "!�
unknown���������`�
E__inference_dense_46_layer_call_and_return_conditional_losses_1804979e��/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������`
� �
*__inference_dense_46_layer_call_fn_1804969Z��/�,
%�"
 �
inputs���������
� "!�
unknown���������`�
E__inference_dense_47_layer_call_and_return_conditional_losses_1805069e��/�,
%�"
 �
inputs���������`
� ",�)
"�
tensor_0���������
� �
*__inference_dense_47_layer_call_fn_1805058Z��/�,
%�"
 �
inputs���������`
� "!�
unknown����������
G__inference_dropout_10_layer_call_and_return_conditional_losses_1805001c3�0
)�&
 �
inputs���������`
p
� ",�)
"�
tensor_0���������`
� �
G__inference_dropout_10_layer_call_and_return_conditional_losses_1805006c3�0
)�&
 �
inputs���������`
p 
� ",�)
"�
tensor_0���������`
� �
,__inference_dropout_10_layer_call_fn_1804984X3�0
)�&
 �
inputs���������`
p
� "!�
unknown���������`�
,__inference_dropout_10_layer_call_fn_1804989X3�0
)�&
 �
inputs���������`
p 
� "!�
unknown���������`�
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804955c3�0
)�&
 �
inputs���������`
p
� ",�)
"�
tensor_0���������`
� �
G__inference_dropout_11_layer_call_and_return_conditional_losses_1804960c3�0
)�&
 �
inputs���������`
p 
� ",�)
"�
tensor_0���������`
� �
,__inference_dropout_11_layer_call_fn_1804938X3�0
)�&
 �
inputs���������`
p
� "!�
unknown���������`�
,__inference_dropout_11_layer_call_fn_1804943X3�0
)�&
 �
inputs���������`
p 
� "!�
unknown���������`�
F__inference_flatten_5_layer_call_and_return_conditional_losses_1804642c3�0
)�&
$�!
inputs���������
� ",�)
"�
tensor_0���������
� �
+__inference_flatten_5_layer_call_fn_1804636X3�0
)�&
$�!
inputs���������
� "!�
unknown����������
K__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_1804715a0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
0__inference_leaky_re_lu_20_layer_call_fn_1804710V0�-
&�#
!�
inputs����������
� ""�
unknown�����������
K__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_1804773_/�,
%�"
 �
inputs���������`
� ",�)
"�
tensor_0���������`
� �
0__inference_leaky_re_lu_21_layer_call_fn_1804768T/�,
%�"
 �
inputs���������`
� "!�
unknown���������`�
K__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_1804705a0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
0__inference_leaky_re_lu_22_layer_call_fn_1804700V0�-
&�#
!�
inputs����������
� ""�
unknown�����������
K__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_1804763_/�,
%�"
 �
inputs���������`
� ",�)
"�
tensor_0���������`
� �
0__inference_leaky_re_lu_23_layer_call_fn_1804758T/�,
%�"
 �
inputs���������`
� "!�
unknown���������`�
D__inference_model_5_layer_call_and_return_conditional_losses_1804132� 56=>QRYZpqno��{|yz���������
���
��~
0�-
pixel_projection_y���������
 �
angles���������
(�%
cluster_charge���������
p

 
� ",�)
"�
tensor_0���������
� �
D__inference_model_5_layer_call_and_return_conditional_losses_1804221� 56=>QRYZqnpo��|y{z���������
���
��~
0�-
pixel_projection_y���������
 �
angles���������
(�%
cluster_charge���������
p 

 
� ",�)
"�
tensor_0���������
� �
)__inference_model_5_layer_call_fn_1804276� 56=>QRYZpqno��{|yz���������
���
��~
0�-
pixel_projection_y���������
 �
angles���������
(�%
cluster_charge���������
p

 
� "!�
unknown����������
)__inference_model_5_layer_call_fn_1804331� 56=>QRYZqnpo��|y{z���������
���
��~
0�-
pixel_projection_y���������
 �
angles���������
(�%
cluster_charge���������
p 

 
� "!�
unknown����������
%__inference_signature_wrapper_1804531� 56=>QRYZqnpo��|y{z���������
� 
���
*
angles �
angles���������
:
cluster_charge(�%
cluster_charge���������
F
pixel_projection_y0�-
pixel_projection_y���������"?�<
:
concatenate_11(�%
concatenate_11���������