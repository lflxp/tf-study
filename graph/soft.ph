
H
input/x_inputPlaceholder*
dtype0*
shape:����������
G
input/y_inputPlaceholder*
shape:���������
*
dtype0
R
layer/W/zeros/shape_as_tensorConst*
valueB"  
   *
dtype0
@
layer/W/zeros/ConstConst*
dtype0*
valueB
 *    
d
layer/W/zerosFilllayer/W/zeros/shape_as_tensorlayer/W/zeros/Const*
T0*

index_type0
d
layer/W/Weights
VariableV2*
dtype0*
	container *
shape:	�
*
shared_name 
�
layer/W/Weights/AssignAssignlayer/W/Weightslayer/W/zeros*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
^
layer/W/Weights/readIdentitylayer/W/Weights*
T0*"
_class
loc:@layer/W/Weights
>
layer/b/zerosConst*
valueB
*    *
dtype0
^
layer/b/biases
VariableV2*
shape:
*
shared_name *
dtype0*
	container 
�
layer/b/biases/AssignAssignlayer/b/biaseslayer/b/zeros*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
[
layer/b/biases/readIdentitylayer/b/biases*
T0*!
_class
loc:@layer/b/biases
p
layer/W_p_b/MatMulMatMulinput/x_inputlayer/W/Weights/read*
transpose_a( *
transpose_b( *
T0
N
layer/W_p_b/Wx_plus_bAddlayer/W_p_b/MatMullayer/b/biases/read*
T0
=
layer/final_resultSoftmaxlayer/W_p_b/Wx_plus_b*
T0
,
loss/LogLoglayer/final_result*
T0
1
loss/mulMulinput/y_inputloss/Log*
T0
?

loss/ConstConst*
dtype0*
valueB"       
K
loss/SumSumloss/mul
loss/Const*

Tidx0*
	keep_dims( *
T0
"
loss/NegNegloss/Sum*
T0
C
train_step/gradients/ShapeConst*
valueB *
dtype0
K
train_step/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0
x
train_step/gradients/FillFilltrain_step/gradients/Shapetrain_step/gradients/grad_ys_0*
T0*

index_type0
Q
&train_step/gradients/loss/Neg_grad/NegNegtrain_step/gradients/Fill*
T0
e
0train_step/gradients/loss/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0
�
*train_step/gradients/loss/Sum_grad/ReshapeReshape&train_step/gradients/loss/Neg_grad/Neg0train_step/gradients/loss/Sum_grad/Reshape/shape*
T0*
Tshape0
T
(train_step/gradients/loss/Sum_grad/ShapeShapeloss/mul*
T0*
out_type0
�
'train_step/gradients/loss/Sum_grad/TileTile*train_step/gradients/loss/Sum_grad/Reshape(train_step/gradients/loss/Sum_grad/Shape*

Tmultiples0*
T0
Y
(train_step/gradients/loss/mul_grad/ShapeShapeinput/y_input*
T0*
out_type0
V
*train_step/gradients/loss/mul_grad/Shape_1Shapeloss/Log*
T0*
out_type0
�
8train_step/gradients/loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(train_step/gradients/loss/mul_grad/Shape*train_step/gradients/loss/mul_grad/Shape_1*
T0
i
&train_step/gradients/loss/mul_grad/MulMul'train_step/gradients/loss/Sum_grad/Tileloss/Log*
T0
�
&train_step/gradients/loss/mul_grad/SumSum&train_step/gradients/loss/mul_grad/Mul8train_step/gradients/loss/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
*train_step/gradients/loss/mul_grad/ReshapeReshape&train_step/gradients/loss/mul_grad/Sum(train_step/gradients/loss/mul_grad/Shape*
T0*
Tshape0
p
(train_step/gradients/loss/mul_grad/Mul_1Mulinput/y_input'train_step/gradients/loss/Sum_grad/Tile*
T0
�
(train_step/gradients/loss/mul_grad/Sum_1Sum(train_step/gradients/loss/mul_grad/Mul_1:train_step/gradients/loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
,train_step/gradients/loss/mul_grad/Reshape_1Reshape(train_step/gradients/loss/mul_grad/Sum_1*train_step/gradients/loss/mul_grad/Shape_1*
T0*
Tshape0
�
3train_step/gradients/loss/mul_grad/tuple/group_depsNoOp+^train_step/gradients/loss/mul_grad/Reshape-^train_step/gradients/loss/mul_grad/Reshape_1
�
;train_step/gradients/loss/mul_grad/tuple/control_dependencyIdentity*train_step/gradients/loss/mul_grad/Reshape4^train_step/gradients/loss/mul_grad/tuple/group_deps*=
_class3
1/loc:@train_step/gradients/loss/mul_grad/Reshape*
T0
�
=train_step/gradients/loss/mul_grad/tuple/control_dependency_1Identity,train_step/gradients/loss/mul_grad/Reshape_14^train_step/gradients/loss/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train_step/gradients/loss/mul_grad/Reshape_1
�
-train_step/gradients/loss/Log_grad/Reciprocal
Reciprocallayer/final_result>^train_step/gradients/loss/mul_grad/tuple/control_dependency_1*
T0
�
&train_step/gradients/loss/Log_grad/mulMul=train_step/gradients/loss/mul_grad/tuple/control_dependency_1-train_step/gradients/loss/Log_grad/Reciprocal*
T0
|
0train_step/gradients/layer/final_result_grad/mulMul&train_step/gradients/loss/Log_grad/mullayer/final_result*
T0
u
Btrain_step/gradients/layer/final_result_grad/Sum/reduction_indicesConst*
valueB :
���������*
dtype0
�
0train_step/gradients/layer/final_result_grad/SumSum0train_step/gradients/layer/final_result_grad/mulBtrain_step/gradients/layer/final_result_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
0train_step/gradients/layer/final_result_grad/subSub&train_step/gradients/loss/Log_grad/mul0train_step/gradients/layer/final_result_grad/Sum*
T0
�
2train_step/gradients/layer/final_result_grad/mul_1Mul0train_step/gradients/layer/final_result_grad/sublayer/final_result*
T0
k
5train_step/gradients/layer/W_p_b/Wx_plus_b_grad/ShapeShapelayer/W_p_b/MatMul*
T0*
out_type0
e
7train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Shape_1Const*
valueB:
*
dtype0
�
Etrain_step/gradients/layer/W_p_b/Wx_plus_b_grad/BroadcastGradientArgsBroadcastGradientArgs5train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Shape7train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Shape_1*
T0
�
3train_step/gradients/layer/W_p_b/Wx_plus_b_grad/SumSum2train_step/gradients/layer/final_result_grad/mul_1Etrain_step/gradients/layer/W_p_b/Wx_plus_b_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
7train_step/gradients/layer/W_p_b/Wx_plus_b_grad/ReshapeReshape3train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Sum5train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Shape*
T0*
Tshape0
�
5train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Sum_1Sum2train_step/gradients/layer/final_result_grad/mul_1Gtrain_step/gradients/layer/W_p_b/Wx_plus_b_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
9train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Reshape_1Reshape5train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Sum_17train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Shape_1*
T0*
Tshape0
�
@train_step/gradients/layer/W_p_b/Wx_plus_b_grad/tuple/group_depsNoOp8^train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Reshape:^train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Reshape_1
�
Htrain_step/gradients/layer/W_p_b/Wx_plus_b_grad/tuple/control_dependencyIdentity7train_step/gradients/layer/W_p_b/Wx_plus_b_grad/ReshapeA^train_step/gradients/layer/W_p_b/Wx_plus_b_grad/tuple/group_deps*
T0*J
_class@
><loc:@train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Reshape
�
Jtrain_step/gradients/layer/W_p_b/Wx_plus_b_grad/tuple/control_dependency_1Identity9train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Reshape_1A^train_step/gradients/layer/W_p_b/Wx_plus_b_grad/tuple/group_deps*L
_classB
@>loc:@train_step/gradients/layer/W_p_b/Wx_plus_b_grad/Reshape_1*
T0
�
3train_step/gradients/layer/W_p_b/MatMul_grad/MatMulMatMulHtrain_step/gradients/layer/W_p_b/Wx_plus_b_grad/tuple/control_dependencylayer/W/Weights/read*
T0*
transpose_a( *
transpose_b(
�
5train_step/gradients/layer/W_p_b/MatMul_grad/MatMul_1MatMulinput/x_inputHtrain_step/gradients/layer/W_p_b/Wx_plus_b_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
�
=train_step/gradients/layer/W_p_b/MatMul_grad/tuple/group_depsNoOp4^train_step/gradients/layer/W_p_b/MatMul_grad/MatMul6^train_step/gradients/layer/W_p_b/MatMul_grad/MatMul_1
�
Etrain_step/gradients/layer/W_p_b/MatMul_grad/tuple/control_dependencyIdentity3train_step/gradients/layer/W_p_b/MatMul_grad/MatMul>^train_step/gradients/layer/W_p_b/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train_step/gradients/layer/W_p_b/MatMul_grad/MatMul
�
Gtrain_step/gradients/layer/W_p_b/MatMul_grad/tuple/control_dependency_1Identity5train_step/gradients/layer/W_p_b/MatMul_grad/MatMul_1>^train_step/gradients/layer/W_p_b/MatMul_grad/tuple/group_deps*H
_class>
<:loc:@train_step/gradients/layer/W_p_b/MatMul_grad/MatMul_1*
T0
U
(train_step/GradientDescent/learning_rateConst*
valueB
 *
�#<*
dtype0
�
Ftrain_step/GradientDescent/update_layer/W/Weights/ApplyGradientDescentApplyGradientDescentlayer/W/Weights(train_step/GradientDescent/learning_rateGtrain_step/gradients/layer/W_p_b/MatMul_grad/tuple/control_dependency_1*"
_class
loc:@layer/W/Weights*
use_locking( *
T0
�
Etrain_step/GradientDescent/update_layer/b/biases/ApplyGradientDescentApplyGradientDescentlayer/b/biases(train_step/GradientDescent/learning_rateJtrain_step/gradients/layer/W_p_b/Wx_plus_b_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@layer/b/biases
�
train_step/GradientDescentNoOpG^train_step/GradientDescent/update_layer/W/Weights/ApplyGradientDescentF^train_step/GradientDescent/update_layer/b/biases/ApplyGradientDescent
=
initNoOp^layer/W/Weights/Assign^layer/b/biases/Assign
8

save/ConstConst*
valueB Bmodel*
dtype0
d
save/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
K
save/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
v
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
]
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save/AssignAssignlayer/W/Weightssave/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save/Assign_1Assignlayer/b/biasessave/RestoreV2:1*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(*
T0
6
save/restore_allNoOp^save/Assign^save/Assign_1
:
save_1/ConstConst*
valueB Bmodel*
dtype0
f
save_1/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
M
save_1/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
m
save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
T0*
_class
loc:@save_1/Const
x
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
_
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_1/AssignAssignlayer/W/Weightssave_1/RestoreV2*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(*
T0
�
save_1/Assign_1Assignlayer/b/biasessave_1/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
<
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1
:
save_2/ConstConst*
valueB Bmodel*
dtype0
f
save_2/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
M
save_2/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_2/SaveV2SaveV2save_2/Constsave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
m
save_2/control_dependencyIdentitysave_2/Const^save_2/SaveV2*
T0*
_class
loc:@save_2/Const
x
save_2/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
_
!save_2/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_2/AssignAssignlayer/W/Weightssave_2/RestoreV2*
validate_shape(*
use_locking(*
T0*"
_class
loc:@layer/W/Weights
�
save_2/Assign_1Assignlayer/b/biasessave_2/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
<
save_2/restore_allNoOp^save_2/Assign^save_2/Assign_1
:
save_3/ConstConst*
valueB Bmodel*
dtype0
f
save_3/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
M
save_3/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_3/SaveV2SaveV2save_3/Constsave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
m
save_3/control_dependencyIdentitysave_3/Const^save_3/SaveV2*
_class
loc:@save_3/Const*
T0
x
save_3/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
_
!save_3/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_3/AssignAssignlayer/W/Weightssave_3/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_3/Assign_1Assignlayer/b/biasessave_3/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
<
save_3/restore_allNoOp^save_3/Assign^save_3/Assign_1
:
save_4/ConstConst*
valueB Bmodel*
dtype0
f
save_4/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
M
save_4/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_4/SaveV2SaveV2save_4/Constsave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
m
save_4/control_dependencyIdentitysave_4/Const^save_4/SaveV2*
T0*
_class
loc:@save_4/Const
x
save_4/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
_
!save_4/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_4/AssignAssignlayer/W/Weightssave_4/RestoreV2*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(*
T0
�
save_4/Assign_1Assignlayer/b/biasessave_4/RestoreV2:1*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(*
T0
<
save_4/restore_allNoOp^save_4/Assign^save_4/Assign_1
:
save_5/ConstConst*
dtype0*
valueB Bmodel
f
save_5/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
M
save_5/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
�
save_5/SaveV2SaveV2save_5/Constsave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
m
save_5/control_dependencyIdentitysave_5/Const^save_5/SaveV2*
T0*
_class
loc:@save_5/Const
x
save_5/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
_
!save_5/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_5/AssignAssignlayer/W/Weightssave_5/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_5/Assign_1Assignlayer/b/biasessave_5/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
<
save_5/restore_allNoOp^save_5/Assign^save_5/Assign_1
:
save_6/ConstConst*
valueB Bmodel*
dtype0
f
save_6/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
M
save_6/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_6/SaveV2SaveV2save_6/Constsave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
m
save_6/control_dependencyIdentitysave_6/Const^save_6/SaveV2*
T0*
_class
loc:@save_6/Const
x
save_6/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
_
!save_6/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_6/AssignAssignlayer/W/Weightssave_6/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_6/Assign_1Assignlayer/b/biasessave_6/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
<
save_6/restore_allNoOp^save_6/Assign^save_6/Assign_1
:
save_7/ConstConst*
valueB Bmodel*
dtype0
f
save_7/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
M
save_7/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_7/SaveV2SaveV2save_7/Constsave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
m
save_7/control_dependencyIdentitysave_7/Const^save_7/SaveV2*
T0*
_class
loc:@save_7/Const
x
save_7/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
_
!save_7/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_7/AssignAssignlayer/W/Weightssave_7/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_7/Assign_1Assignlayer/b/biasessave_7/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
<
save_7/restore_allNoOp^save_7/Assign^save_7/Assign_1
:
save_8/ConstConst*
valueB Bmodel*
dtype0
f
save_8/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
M
save_8/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_8/SaveV2SaveV2save_8/Constsave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
m
save_8/control_dependencyIdentitysave_8/Const^save_8/SaveV2*
T0*
_class
loc:@save_8/Const
x
save_8/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
_
!save_8/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_8/AssignAssignlayer/W/Weightssave_8/RestoreV2*
validate_shape(*
use_locking(*
T0*"
_class
loc:@layer/W/Weights
�
save_8/Assign_1Assignlayer/b/biasessave_8/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
<
save_8/restore_allNoOp^save_8/Assign^save_8/Assign_1
:
save_9/ConstConst*
valueB Bmodel*
dtype0
f
save_9/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
M
save_9/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_9/SaveV2SaveV2save_9/Constsave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
m
save_9/control_dependencyIdentitysave_9/Const^save_9/SaveV2*
T0*
_class
loc:@save_9/Const
x
save_9/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
_
!save_9/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_9/AssignAssignlayer/W/Weightssave_9/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_9/Assign_1Assignlayer/b/biasessave_9/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
<
save_9/restore_allNoOp^save_9/Assign^save_9/Assign_1
;
save_10/ConstConst*
valueB Bmodel*
dtype0
g
save_10/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_10/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_10/SaveV2SaveV2save_10/Constsave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_10/control_dependencyIdentitysave_10/Const^save_10/SaveV2*
T0* 
_class
loc:@save_10/Const
y
save_10/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_10/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_10/AssignAssignlayer/W/Weightssave_10/RestoreV2*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(*
T0
�
save_10/Assign_1Assignlayer/b/biasessave_10/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_10/restore_allNoOp^save_10/Assign^save_10/Assign_1
;
save_11/ConstConst*
valueB Bmodel*
dtype0
g
save_11/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_11/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_11/SaveV2SaveV2save_11/Constsave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_11/control_dependencyIdentitysave_11/Const^save_11/SaveV2*
T0* 
_class
loc:@save_11/Const
y
save_11/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_11/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_11/AssignAssignlayer/W/Weightssave_11/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_11/Assign_1Assignlayer/b/biasessave_11/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_11/restore_allNoOp^save_11/Assign^save_11/Assign_1
;
save_12/ConstConst*
valueB Bmodel*
dtype0
g
save_12/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_12/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_12/SaveV2SaveV2save_12/Constsave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_12/control_dependencyIdentitysave_12/Const^save_12/SaveV2*
T0* 
_class
loc:@save_12/Const
y
save_12/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_12/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_12/AssignAssignlayer/W/Weightssave_12/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_12/Assign_1Assignlayer/b/biasessave_12/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_12/restore_allNoOp^save_12/Assign^save_12/Assign_1
;
save_13/ConstConst*
valueB Bmodel*
dtype0
g
save_13/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_13/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_13/SaveV2SaveV2save_13/Constsave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_13/control_dependencyIdentitysave_13/Const^save_13/SaveV2*
T0* 
_class
loc:@save_13/Const
y
save_13/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_13/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_13/AssignAssignlayer/W/Weightssave_13/RestoreV2*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(*
T0
�
save_13/Assign_1Assignlayer/b/biasessave_13/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_13/restore_allNoOp^save_13/Assign^save_13/Assign_1
;
save_14/ConstConst*
valueB Bmodel*
dtype0
g
save_14/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_14/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_14/SaveV2SaveV2save_14/Constsave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_14/control_dependencyIdentitysave_14/Const^save_14/SaveV2*
T0* 
_class
loc:@save_14/Const
y
save_14/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_14/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_14/AssignAssignlayer/W/Weightssave_14/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_14/Assign_1Assignlayer/b/biasessave_14/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_14/restore_allNoOp^save_14/Assign^save_14/Assign_1
;
save_15/ConstConst*
valueB Bmodel*
dtype0
g
save_15/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_15/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_15/SaveV2SaveV2save_15/Constsave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_15/control_dependencyIdentitysave_15/Const^save_15/SaveV2*
T0* 
_class
loc:@save_15/Const
y
save_15/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_15/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_15/AssignAssignlayer/W/Weightssave_15/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_15/Assign_1Assignlayer/b/biasessave_15/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_15/restore_allNoOp^save_15/Assign^save_15/Assign_1
;
save_16/ConstConst*
valueB Bmodel*
dtype0
g
save_16/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_16/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_16/SaveV2SaveV2save_16/Constsave_16/SaveV2/tensor_namessave_16/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_16/control_dependencyIdentitysave_16/Const^save_16/SaveV2*
T0* 
_class
loc:@save_16/Const
y
save_16/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_16/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_16/RestoreV2	RestoreV2save_16/Constsave_16/RestoreV2/tensor_names"save_16/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_16/AssignAssignlayer/W/Weightssave_16/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_16/Assign_1Assignlayer/b/biasessave_16/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_16/restore_allNoOp^save_16/Assign^save_16/Assign_1
;
save_17/ConstConst*
dtype0*
valueB Bmodel
g
save_17/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_17/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
�
save_17/SaveV2SaveV2save_17/Constsave_17/SaveV2/tensor_namessave_17/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_17/control_dependencyIdentitysave_17/Const^save_17/SaveV2*
T0* 
_class
loc:@save_17/Const
y
save_17/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_17/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_17/RestoreV2	RestoreV2save_17/Constsave_17/RestoreV2/tensor_names"save_17/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_17/AssignAssignlayer/W/Weightssave_17/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_17/Assign_1Assignlayer/b/biasessave_17/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_17/restore_allNoOp^save_17/Assign^save_17/Assign_1
;
save_18/ConstConst*
valueB Bmodel*
dtype0
g
save_18/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_18/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_18/SaveV2SaveV2save_18/Constsave_18/SaveV2/tensor_namessave_18/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_18/control_dependencyIdentitysave_18/Const^save_18/SaveV2*
T0* 
_class
loc:@save_18/Const
y
save_18/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_18/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_18/RestoreV2	RestoreV2save_18/Constsave_18/RestoreV2/tensor_names"save_18/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_18/AssignAssignlayer/W/Weightssave_18/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_18/Assign_1Assignlayer/b/biasessave_18/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_18/restore_allNoOp^save_18/Assign^save_18/Assign_1
;
save_19/ConstConst*
valueB Bmodel*
dtype0
g
save_19/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_19/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_19/SaveV2SaveV2save_19/Constsave_19/SaveV2/tensor_namessave_19/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_19/control_dependencyIdentitysave_19/Const^save_19/SaveV2*
T0* 
_class
loc:@save_19/Const
y
save_19/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_19/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_19/RestoreV2	RestoreV2save_19/Constsave_19/RestoreV2/tensor_names"save_19/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_19/AssignAssignlayer/W/Weightssave_19/RestoreV2*
validate_shape(*
use_locking(*
T0*"
_class
loc:@layer/W/Weights
�
save_19/Assign_1Assignlayer/b/biasessave_19/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_19/restore_allNoOp^save_19/Assign^save_19/Assign_1
;
save_20/ConstConst*
valueB Bmodel*
dtype0
g
save_20/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_20/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_20/SaveV2SaveV2save_20/Constsave_20/SaveV2/tensor_namessave_20/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_20/control_dependencyIdentitysave_20/Const^save_20/SaveV2*
T0* 
_class
loc:@save_20/Const
y
save_20/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_20/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_20/RestoreV2	RestoreV2save_20/Constsave_20/RestoreV2/tensor_names"save_20/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_20/AssignAssignlayer/W/Weightssave_20/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_20/Assign_1Assignlayer/b/biasessave_20/RestoreV2:1*
validate_shape(*
use_locking(*
T0*!
_class
loc:@layer/b/biases
?
save_20/restore_allNoOp^save_20/Assign^save_20/Assign_1
;
save_21/ConstConst*
valueB Bmodel*
dtype0
g
save_21/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_21/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_21/SaveV2SaveV2save_21/Constsave_21/SaveV2/tensor_namessave_21/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_21/control_dependencyIdentitysave_21/Const^save_21/SaveV2*
T0* 
_class
loc:@save_21/Const
y
save_21/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_21/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_21/RestoreV2	RestoreV2save_21/Constsave_21/RestoreV2/tensor_names"save_21/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_21/AssignAssignlayer/W/Weightssave_21/RestoreV2*
validate_shape(*
use_locking(*
T0*"
_class
loc:@layer/W/Weights
�
save_21/Assign_1Assignlayer/b/biasessave_21/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_21/restore_allNoOp^save_21/Assign^save_21/Assign_1
;
save_22/ConstConst*
valueB Bmodel*
dtype0
g
save_22/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_22/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_22/SaveV2SaveV2save_22/Constsave_22/SaveV2/tensor_namessave_22/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_22/control_dependencyIdentitysave_22/Const^save_22/SaveV2*
T0* 
_class
loc:@save_22/Const
y
save_22/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_22/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_22/RestoreV2	RestoreV2save_22/Constsave_22/RestoreV2/tensor_names"save_22/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_22/AssignAssignlayer/W/Weightssave_22/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_22/Assign_1Assignlayer/b/biasessave_22/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_22/restore_allNoOp^save_22/Assign^save_22/Assign_1
;
save_23/ConstConst*
valueB Bmodel*
dtype0
g
save_23/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_23/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_23/SaveV2SaveV2save_23/Constsave_23/SaveV2/tensor_namessave_23/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_23/control_dependencyIdentitysave_23/Const^save_23/SaveV2*
T0* 
_class
loc:@save_23/Const
y
save_23/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_23/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_23/RestoreV2	RestoreV2save_23/Constsave_23/RestoreV2/tensor_names"save_23/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_23/AssignAssignlayer/W/Weightssave_23/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_23/Assign_1Assignlayer/b/biasessave_23/RestoreV2:1*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(*
T0
?
save_23/restore_allNoOp^save_23/Assign^save_23/Assign_1
;
save_24/ConstConst*
valueB Bmodel*
dtype0
g
save_24/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_24/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_24/SaveV2SaveV2save_24/Constsave_24/SaveV2/tensor_namessave_24/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_24/control_dependencyIdentitysave_24/Const^save_24/SaveV2*
T0* 
_class
loc:@save_24/Const
y
save_24/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_24/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_24/RestoreV2	RestoreV2save_24/Constsave_24/RestoreV2/tensor_names"save_24/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_24/AssignAssignlayer/W/Weightssave_24/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_24/Assign_1Assignlayer/b/biasessave_24/RestoreV2:1*
validate_shape(*
use_locking(*
T0*!
_class
loc:@layer/b/biases
?
save_24/restore_allNoOp^save_24/Assign^save_24/Assign_1
;
save_25/ConstConst*
valueB Bmodel*
dtype0
g
save_25/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_25/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_25/SaveV2SaveV2save_25/Constsave_25/SaveV2/tensor_namessave_25/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_25/control_dependencyIdentitysave_25/Const^save_25/SaveV2*
T0* 
_class
loc:@save_25/Const
y
save_25/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_25/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_25/RestoreV2	RestoreV2save_25/Constsave_25/RestoreV2/tensor_names"save_25/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_25/AssignAssignlayer/W/Weightssave_25/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_25/Assign_1Assignlayer/b/biasessave_25/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_25/restore_allNoOp^save_25/Assign^save_25/Assign_1
;
save_26/ConstConst*
dtype0*
valueB Bmodel
g
save_26/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_26/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_26/SaveV2SaveV2save_26/Constsave_26/SaveV2/tensor_namessave_26/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_26/control_dependencyIdentitysave_26/Const^save_26/SaveV2* 
_class
loc:@save_26/Const*
T0
y
save_26/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_26/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_26/RestoreV2	RestoreV2save_26/Constsave_26/RestoreV2/tensor_names"save_26/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_26/AssignAssignlayer/W/Weightssave_26/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_26/Assign_1Assignlayer/b/biasessave_26/RestoreV2:1*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(*
T0
?
save_26/restore_allNoOp^save_26/Assign^save_26/Assign_1
;
save_27/ConstConst*
valueB Bmodel*
dtype0
g
save_27/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_27/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_27/SaveV2SaveV2save_27/Constsave_27/SaveV2/tensor_namessave_27/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_27/control_dependencyIdentitysave_27/Const^save_27/SaveV2*
T0* 
_class
loc:@save_27/Const
y
save_27/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_27/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_27/RestoreV2	RestoreV2save_27/Constsave_27/RestoreV2/tensor_names"save_27/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_27/AssignAssignlayer/W/Weightssave_27/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_27/Assign_1Assignlayer/b/biasessave_27/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_27/restore_allNoOp^save_27/Assign^save_27/Assign_1
;
save_28/ConstConst*
valueB Bmodel*
dtype0
g
save_28/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_28/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
�
save_28/SaveV2SaveV2save_28/Constsave_28/SaveV2/tensor_namessave_28/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_28/control_dependencyIdentitysave_28/Const^save_28/SaveV2*
T0* 
_class
loc:@save_28/Const
y
save_28/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_28/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_28/RestoreV2	RestoreV2save_28/Constsave_28/RestoreV2/tensor_names"save_28/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_28/AssignAssignlayer/W/Weightssave_28/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_28/Assign_1Assignlayer/b/biasessave_28/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_28/restore_allNoOp^save_28/Assign^save_28/Assign_1
;
save_29/ConstConst*
valueB Bmodel*
dtype0
g
save_29/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_29/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_29/SaveV2SaveV2save_29/Constsave_29/SaveV2/tensor_namessave_29/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_29/control_dependencyIdentitysave_29/Const^save_29/SaveV2*
T0* 
_class
loc:@save_29/Const
y
save_29/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_29/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_29/RestoreV2	RestoreV2save_29/Constsave_29/RestoreV2/tensor_names"save_29/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_29/AssignAssignlayer/W/Weightssave_29/RestoreV2*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(*
T0
�
save_29/Assign_1Assignlayer/b/biasessave_29/RestoreV2:1*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(*
T0
?
save_29/restore_allNoOp^save_29/Assign^save_29/Assign_1
;
save_30/ConstConst*
valueB Bmodel*
dtype0
g
save_30/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_30/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_30/SaveV2SaveV2save_30/Constsave_30/SaveV2/tensor_namessave_30/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_30/control_dependencyIdentitysave_30/Const^save_30/SaveV2*
T0* 
_class
loc:@save_30/Const
y
save_30/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_30/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_30/RestoreV2	RestoreV2save_30/Constsave_30/RestoreV2/tensor_names"save_30/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_30/AssignAssignlayer/W/Weightssave_30/RestoreV2*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(*
T0
�
save_30/Assign_1Assignlayer/b/biasessave_30/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_30/restore_allNoOp^save_30/Assign^save_30/Assign_1
;
save_31/ConstConst*
valueB Bmodel*
dtype0
g
save_31/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_31/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_31/SaveV2SaveV2save_31/Constsave_31/SaveV2/tensor_namessave_31/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_31/control_dependencyIdentitysave_31/Const^save_31/SaveV2*
T0* 
_class
loc:@save_31/Const
y
save_31/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_31/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_31/RestoreV2	RestoreV2save_31/Constsave_31/RestoreV2/tensor_names"save_31/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_31/AssignAssignlayer/W/Weightssave_31/RestoreV2*
validate_shape(*
use_locking(*
T0*"
_class
loc:@layer/W/Weights
�
save_31/Assign_1Assignlayer/b/biasessave_31/RestoreV2:1*
validate_shape(*
use_locking(*
T0*!
_class
loc:@layer/b/biases
?
save_31/restore_allNoOp^save_31/Assign^save_31/Assign_1
;
save_32/ConstConst*
valueB Bmodel*
dtype0
g
save_32/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_32/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_32/SaveV2SaveV2save_32/Constsave_32/SaveV2/tensor_namessave_32/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_32/control_dependencyIdentitysave_32/Const^save_32/SaveV2*
T0* 
_class
loc:@save_32/Const
y
save_32/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_32/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_32/RestoreV2	RestoreV2save_32/Constsave_32/RestoreV2/tensor_names"save_32/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_32/AssignAssignlayer/W/Weightssave_32/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_32/Assign_1Assignlayer/b/biasessave_32/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_32/restore_allNoOp^save_32/Assign^save_32/Assign_1
;
save_33/ConstConst*
valueB Bmodel*
dtype0
g
save_33/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_33/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_33/SaveV2SaveV2save_33/Constsave_33/SaveV2/tensor_namessave_33/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_33/control_dependencyIdentitysave_33/Const^save_33/SaveV2*
T0* 
_class
loc:@save_33/Const
y
save_33/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_33/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_33/RestoreV2	RestoreV2save_33/Constsave_33/RestoreV2/tensor_names"save_33/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_33/AssignAssignlayer/W/Weightssave_33/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_33/Assign_1Assignlayer/b/biasessave_33/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_33/restore_allNoOp^save_33/Assign^save_33/Assign_1
;
save_34/ConstConst*
dtype0*
valueB Bmodel
g
save_34/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_34/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_34/SaveV2SaveV2save_34/Constsave_34/SaveV2/tensor_namessave_34/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_34/control_dependencyIdentitysave_34/Const^save_34/SaveV2*
T0* 
_class
loc:@save_34/Const
y
save_34/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_34/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_34/RestoreV2	RestoreV2save_34/Constsave_34/RestoreV2/tensor_names"save_34/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_34/AssignAssignlayer/W/Weightssave_34/RestoreV2*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(*
T0
�
save_34/Assign_1Assignlayer/b/biasessave_34/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_34/restore_allNoOp^save_34/Assign^save_34/Assign_1
;
save_35/ConstConst*
valueB Bmodel*
dtype0
g
save_35/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_35/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_35/SaveV2SaveV2save_35/Constsave_35/SaveV2/tensor_namessave_35/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_35/control_dependencyIdentitysave_35/Const^save_35/SaveV2*
T0* 
_class
loc:@save_35/Const
y
save_35/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_35/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_35/RestoreV2	RestoreV2save_35/Constsave_35/RestoreV2/tensor_names"save_35/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_35/AssignAssignlayer/W/Weightssave_35/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_35/Assign_1Assignlayer/b/biasessave_35/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_35/restore_allNoOp^save_35/Assign^save_35/Assign_1
;
save_36/ConstConst*
valueB Bmodel*
dtype0
g
save_36/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_36/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_36/SaveV2SaveV2save_36/Constsave_36/SaveV2/tensor_namessave_36/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_36/control_dependencyIdentitysave_36/Const^save_36/SaveV2*
T0* 
_class
loc:@save_36/Const
y
save_36/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_36/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_36/RestoreV2	RestoreV2save_36/Constsave_36/RestoreV2/tensor_names"save_36/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_36/AssignAssignlayer/W/Weightssave_36/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_36/Assign_1Assignlayer/b/biasessave_36/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_36/restore_allNoOp^save_36/Assign^save_36/Assign_1
;
save_37/ConstConst*
valueB Bmodel*
dtype0
g
save_37/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_37/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_37/SaveV2SaveV2save_37/Constsave_37/SaveV2/tensor_namessave_37/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_37/control_dependencyIdentitysave_37/Const^save_37/SaveV2*
T0* 
_class
loc:@save_37/Const
y
save_37/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_37/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_37/RestoreV2	RestoreV2save_37/Constsave_37/RestoreV2/tensor_names"save_37/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_37/AssignAssignlayer/W/Weightssave_37/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_37/Assign_1Assignlayer/b/biasessave_37/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_37/restore_allNoOp^save_37/Assign^save_37/Assign_1
;
save_38/ConstConst*
valueB Bmodel*
dtype0
g
save_38/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_38/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_38/SaveV2SaveV2save_38/Constsave_38/SaveV2/tensor_namessave_38/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_38/control_dependencyIdentitysave_38/Const^save_38/SaveV2*
T0* 
_class
loc:@save_38/Const
y
save_38/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_38/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_38/RestoreV2	RestoreV2save_38/Constsave_38/RestoreV2/tensor_names"save_38/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_38/AssignAssignlayer/W/Weightssave_38/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_38/Assign_1Assignlayer/b/biasessave_38/RestoreV2:1*
validate_shape(*
use_locking(*
T0*!
_class
loc:@layer/b/biases
?
save_38/restore_allNoOp^save_38/Assign^save_38/Assign_1
;
save_39/ConstConst*
valueB Bmodel*
dtype0
g
save_39/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_39/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_39/SaveV2SaveV2save_39/Constsave_39/SaveV2/tensor_namessave_39/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_39/control_dependencyIdentitysave_39/Const^save_39/SaveV2*
T0* 
_class
loc:@save_39/Const
y
save_39/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_39/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_39/RestoreV2	RestoreV2save_39/Constsave_39/RestoreV2/tensor_names"save_39/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_39/AssignAssignlayer/W/Weightssave_39/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_39/Assign_1Assignlayer/b/biasessave_39/RestoreV2:1*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(*
T0
?
save_39/restore_allNoOp^save_39/Assign^save_39/Assign_1
;
save_40/ConstConst*
valueB Bmodel*
dtype0
g
save_40/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_40/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_40/SaveV2SaveV2save_40/Constsave_40/SaveV2/tensor_namessave_40/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_40/control_dependencyIdentitysave_40/Const^save_40/SaveV2*
T0* 
_class
loc:@save_40/Const
y
save_40/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_40/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_40/RestoreV2	RestoreV2save_40/Constsave_40/RestoreV2/tensor_names"save_40/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_40/AssignAssignlayer/W/Weightssave_40/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_40/Assign_1Assignlayer/b/biasessave_40/RestoreV2:1*
validate_shape(*
use_locking(*
T0*!
_class
loc:@layer/b/biases
?
save_40/restore_allNoOp^save_40/Assign^save_40/Assign_1
;
save_41/ConstConst*
valueB Bmodel*
dtype0
g
save_41/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_41/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_41/SaveV2SaveV2save_41/Constsave_41/SaveV2/tensor_namessave_41/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_41/control_dependencyIdentitysave_41/Const^save_41/SaveV2*
T0* 
_class
loc:@save_41/Const
y
save_41/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_41/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_41/RestoreV2	RestoreV2save_41/Constsave_41/RestoreV2/tensor_names"save_41/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_41/AssignAssignlayer/W/Weightssave_41/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_41/Assign_1Assignlayer/b/biasessave_41/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_41/restore_allNoOp^save_41/Assign^save_41/Assign_1
;
save_42/ConstConst*
dtype0*
valueB Bmodel
g
save_42/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_42/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_42/SaveV2SaveV2save_42/Constsave_42/SaveV2/tensor_namessave_42/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_42/control_dependencyIdentitysave_42/Const^save_42/SaveV2* 
_class
loc:@save_42/Const*
T0
y
save_42/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_42/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_42/RestoreV2	RestoreV2save_42/Constsave_42/RestoreV2/tensor_names"save_42/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_42/AssignAssignlayer/W/Weightssave_42/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_42/Assign_1Assignlayer/b/biasessave_42/RestoreV2:1*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(*
T0
?
save_42/restore_allNoOp^save_42/Assign^save_42/Assign_1
;
save_43/ConstConst*
valueB Bmodel*
dtype0
g
save_43/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_43/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_43/SaveV2SaveV2save_43/Constsave_43/SaveV2/tensor_namessave_43/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_43/control_dependencyIdentitysave_43/Const^save_43/SaveV2*
T0* 
_class
loc:@save_43/Const
y
save_43/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_43/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_43/RestoreV2	RestoreV2save_43/Constsave_43/RestoreV2/tensor_names"save_43/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_43/AssignAssignlayer/W/Weightssave_43/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_43/Assign_1Assignlayer/b/biasessave_43/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_43/restore_allNoOp^save_43/Assign^save_43/Assign_1
;
save_44/ConstConst*
valueB Bmodel*
dtype0
g
save_44/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_44/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_44/SaveV2SaveV2save_44/Constsave_44/SaveV2/tensor_namessave_44/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_44/control_dependencyIdentitysave_44/Const^save_44/SaveV2* 
_class
loc:@save_44/Const*
T0
y
save_44/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_44/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_44/RestoreV2	RestoreV2save_44/Constsave_44/RestoreV2/tensor_names"save_44/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_44/AssignAssignlayer/W/Weightssave_44/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_44/Assign_1Assignlayer/b/biasessave_44/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_44/restore_allNoOp^save_44/Assign^save_44/Assign_1
;
save_45/ConstConst*
valueB Bmodel*
dtype0
g
save_45/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_45/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_45/SaveV2SaveV2save_45/Constsave_45/SaveV2/tensor_namessave_45/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_45/control_dependencyIdentitysave_45/Const^save_45/SaveV2*
T0* 
_class
loc:@save_45/Const
y
save_45/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_45/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_45/RestoreV2	RestoreV2save_45/Constsave_45/RestoreV2/tensor_names"save_45/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_45/AssignAssignlayer/W/Weightssave_45/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_45/Assign_1Assignlayer/b/biasessave_45/RestoreV2:1*
validate_shape(*
use_locking(*
T0*!
_class
loc:@layer/b/biases
?
save_45/restore_allNoOp^save_45/Assign^save_45/Assign_1
;
save_46/ConstConst*
valueB Bmodel*
dtype0
g
save_46/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_46/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_46/SaveV2SaveV2save_46/Constsave_46/SaveV2/tensor_namessave_46/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_46/control_dependencyIdentitysave_46/Const^save_46/SaveV2*
T0* 
_class
loc:@save_46/Const
y
save_46/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_46/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_46/RestoreV2	RestoreV2save_46/Constsave_46/RestoreV2/tensor_names"save_46/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_46/AssignAssignlayer/W/Weightssave_46/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_46/Assign_1Assignlayer/b/biasessave_46/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_46/restore_allNoOp^save_46/Assign^save_46/Assign_1
;
save_47/ConstConst*
valueB Bmodel*
dtype0
g
save_47/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_47/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_47/SaveV2SaveV2save_47/Constsave_47/SaveV2/tensor_namessave_47/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_47/control_dependencyIdentitysave_47/Const^save_47/SaveV2*
T0* 
_class
loc:@save_47/Const
y
save_47/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_47/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_47/RestoreV2	RestoreV2save_47/Constsave_47/RestoreV2/tensor_names"save_47/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_47/AssignAssignlayer/W/Weightssave_47/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_47/Assign_1Assignlayer/b/biasessave_47/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_47/restore_allNoOp^save_47/Assign^save_47/Assign_1
;
save_48/ConstConst*
valueB Bmodel*
dtype0
g
save_48/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_48/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_48/SaveV2SaveV2save_48/Constsave_48/SaveV2/tensor_namessave_48/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_48/control_dependencyIdentitysave_48/Const^save_48/SaveV2*
T0* 
_class
loc:@save_48/Const
y
save_48/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_48/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_48/RestoreV2	RestoreV2save_48/Constsave_48/RestoreV2/tensor_names"save_48/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_48/AssignAssignlayer/W/Weightssave_48/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_48/Assign_1Assignlayer/b/biasessave_48/RestoreV2:1*
validate_shape(*
use_locking(*
T0*!
_class
loc:@layer/b/biases
?
save_48/restore_allNoOp^save_48/Assign^save_48/Assign_1
;
save_49/ConstConst*
dtype0*
valueB Bmodel
g
save_49/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_49/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_49/SaveV2SaveV2save_49/Constsave_49/SaveV2/tensor_namessave_49/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_49/control_dependencyIdentitysave_49/Const^save_49/SaveV2*
T0* 
_class
loc:@save_49/Const
y
save_49/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_49/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_49/RestoreV2	RestoreV2save_49/Constsave_49/RestoreV2/tensor_names"save_49/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_49/AssignAssignlayer/W/Weightssave_49/RestoreV2*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(*
T0
�
save_49/Assign_1Assignlayer/b/biasessave_49/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_49/restore_allNoOp^save_49/Assign^save_49/Assign_1
;
save_50/ConstConst*
valueB Bmodel*
dtype0
g
save_50/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_50/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_50/SaveV2SaveV2save_50/Constsave_50/SaveV2/tensor_namessave_50/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_50/control_dependencyIdentitysave_50/Const^save_50/SaveV2*
T0* 
_class
loc:@save_50/Const
y
save_50/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_50/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_50/RestoreV2	RestoreV2save_50/Constsave_50/RestoreV2/tensor_names"save_50/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_50/AssignAssignlayer/W/Weightssave_50/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_50/Assign_1Assignlayer/b/biasessave_50/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_50/restore_allNoOp^save_50/Assign^save_50/Assign_1
;
save_51/ConstConst*
valueB Bmodel*
dtype0
g
save_51/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_51/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_51/SaveV2SaveV2save_51/Constsave_51/SaveV2/tensor_namessave_51/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_51/control_dependencyIdentitysave_51/Const^save_51/SaveV2*
T0* 
_class
loc:@save_51/Const
y
save_51/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_51/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_51/RestoreV2	RestoreV2save_51/Constsave_51/RestoreV2/tensor_names"save_51/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_51/AssignAssignlayer/W/Weightssave_51/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_51/Assign_1Assignlayer/b/biasessave_51/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_51/restore_allNoOp^save_51/Assign^save_51/Assign_1
;
save_52/ConstConst*
valueB Bmodel*
dtype0
g
save_52/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_52/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_52/SaveV2SaveV2save_52/Constsave_52/SaveV2/tensor_namessave_52/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_52/control_dependencyIdentitysave_52/Const^save_52/SaveV2*
T0* 
_class
loc:@save_52/Const
y
save_52/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_52/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_52/RestoreV2	RestoreV2save_52/Constsave_52/RestoreV2/tensor_names"save_52/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_52/AssignAssignlayer/W/Weightssave_52/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_52/Assign_1Assignlayer/b/biasessave_52/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_52/restore_allNoOp^save_52/Assign^save_52/Assign_1
;
save_53/ConstConst*
valueB Bmodel*
dtype0
g
save_53/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_53/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_53/SaveV2SaveV2save_53/Constsave_53/SaveV2/tensor_namessave_53/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_53/control_dependencyIdentitysave_53/Const^save_53/SaveV2*
T0* 
_class
loc:@save_53/Const
y
save_53/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_53/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_53/RestoreV2	RestoreV2save_53/Constsave_53/RestoreV2/tensor_names"save_53/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_53/AssignAssignlayer/W/Weightssave_53/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_53/Assign_1Assignlayer/b/biasessave_53/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_53/restore_allNoOp^save_53/Assign^save_53/Assign_1
;
save_54/ConstConst*
valueB Bmodel*
dtype0
g
save_54/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_54/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_54/SaveV2SaveV2save_54/Constsave_54/SaveV2/tensor_namessave_54/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_54/control_dependencyIdentitysave_54/Const^save_54/SaveV2*
T0* 
_class
loc:@save_54/Const
y
save_54/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_54/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_54/RestoreV2	RestoreV2save_54/Constsave_54/RestoreV2/tensor_names"save_54/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_54/AssignAssignlayer/W/Weightssave_54/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_54/Assign_1Assignlayer/b/biasessave_54/RestoreV2:1*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(*
T0
?
save_54/restore_allNoOp^save_54/Assign^save_54/Assign_1
;
save_55/ConstConst*
valueB Bmodel*
dtype0
g
save_55/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_55/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_55/SaveV2SaveV2save_55/Constsave_55/SaveV2/tensor_namessave_55/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_55/control_dependencyIdentitysave_55/Const^save_55/SaveV2*
T0* 
_class
loc:@save_55/Const
y
save_55/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_55/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_55/RestoreV2	RestoreV2save_55/Constsave_55/RestoreV2/tensor_names"save_55/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_55/AssignAssignlayer/W/Weightssave_55/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_55/Assign_1Assignlayer/b/biasessave_55/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_55/restore_allNoOp^save_55/Assign^save_55/Assign_1
;
save_56/ConstConst*
valueB Bmodel*
dtype0
g
save_56/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_56/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_56/SaveV2SaveV2save_56/Constsave_56/SaveV2/tensor_namessave_56/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_56/control_dependencyIdentitysave_56/Const^save_56/SaveV2*
T0* 
_class
loc:@save_56/Const
y
save_56/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_56/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_56/RestoreV2	RestoreV2save_56/Constsave_56/RestoreV2/tensor_names"save_56/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_56/AssignAssignlayer/W/Weightssave_56/RestoreV2*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(*
T0
�
save_56/Assign_1Assignlayer/b/biasessave_56/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_56/restore_allNoOp^save_56/Assign^save_56/Assign_1
;
save_57/ConstConst*
valueB Bmodel*
dtype0
g
save_57/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_57/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_57/SaveV2SaveV2save_57/Constsave_57/SaveV2/tensor_namessave_57/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_57/control_dependencyIdentitysave_57/Const^save_57/SaveV2*
T0* 
_class
loc:@save_57/Const
y
save_57/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_57/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_57/RestoreV2	RestoreV2save_57/Constsave_57/RestoreV2/tensor_names"save_57/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_57/AssignAssignlayer/W/Weightssave_57/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_57/Assign_1Assignlayer/b/biasessave_57/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_57/restore_allNoOp^save_57/Assign^save_57/Assign_1
;
save_58/ConstConst*
valueB Bmodel*
dtype0
g
save_58/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_58/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_58/SaveV2SaveV2save_58/Constsave_58/SaveV2/tensor_namessave_58/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_58/control_dependencyIdentitysave_58/Const^save_58/SaveV2*
T0* 
_class
loc:@save_58/Const
y
save_58/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_58/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_58/RestoreV2	RestoreV2save_58/Constsave_58/RestoreV2/tensor_names"save_58/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_58/AssignAssignlayer/W/Weightssave_58/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_58/Assign_1Assignlayer/b/biasessave_58/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_58/restore_allNoOp^save_58/Assign^save_58/Assign_1
;
save_59/ConstConst*
valueB Bmodel*
dtype0
g
save_59/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_59/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_59/SaveV2SaveV2save_59/Constsave_59/SaveV2/tensor_namessave_59/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_59/control_dependencyIdentitysave_59/Const^save_59/SaveV2*
T0* 
_class
loc:@save_59/Const
y
save_59/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_59/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_59/RestoreV2	RestoreV2save_59/Constsave_59/RestoreV2/tensor_names"save_59/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_59/AssignAssignlayer/W/Weightssave_59/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_59/Assign_1Assignlayer/b/biasessave_59/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_59/restore_allNoOp^save_59/Assign^save_59/Assign_1
;
save_60/ConstConst*
valueB Bmodel*
dtype0
g
save_60/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_60/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_60/SaveV2SaveV2save_60/Constsave_60/SaveV2/tensor_namessave_60/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_60/control_dependencyIdentitysave_60/Const^save_60/SaveV2*
T0* 
_class
loc:@save_60/Const
y
save_60/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_60/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_60/RestoreV2	RestoreV2save_60/Constsave_60/RestoreV2/tensor_names"save_60/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_60/AssignAssignlayer/W/Weightssave_60/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_60/Assign_1Assignlayer/b/biasessave_60/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_60/restore_allNoOp^save_60/Assign^save_60/Assign_1
;
save_61/ConstConst*
valueB Bmodel*
dtype0
g
save_61/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_61/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_61/SaveV2SaveV2save_61/Constsave_61/SaveV2/tensor_namessave_61/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_61/control_dependencyIdentitysave_61/Const^save_61/SaveV2*
T0* 
_class
loc:@save_61/Const
y
save_61/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_61/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_61/RestoreV2	RestoreV2save_61/Constsave_61/RestoreV2/tensor_names"save_61/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_61/AssignAssignlayer/W/Weightssave_61/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_61/Assign_1Assignlayer/b/biasessave_61/RestoreV2:1*
validate_shape(*
use_locking(*
T0*!
_class
loc:@layer/b/biases
?
save_61/restore_allNoOp^save_61/Assign^save_61/Assign_1
;
save_62/ConstConst*
dtype0*
valueB Bmodel
g
save_62/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_62/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_62/SaveV2SaveV2save_62/Constsave_62/SaveV2/tensor_namessave_62/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_62/control_dependencyIdentitysave_62/Const^save_62/SaveV2*
T0* 
_class
loc:@save_62/Const
y
save_62/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_62/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_62/RestoreV2	RestoreV2save_62/Constsave_62/RestoreV2/tensor_names"save_62/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_62/AssignAssignlayer/W/Weightssave_62/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_62/Assign_1Assignlayer/b/biasessave_62/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_62/restore_allNoOp^save_62/Assign^save_62/Assign_1
;
save_63/ConstConst*
valueB Bmodel*
dtype0
g
save_63/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_63/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_63/SaveV2SaveV2save_63/Constsave_63/SaveV2/tensor_namessave_63/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_63/control_dependencyIdentitysave_63/Const^save_63/SaveV2*
T0* 
_class
loc:@save_63/Const
y
save_63/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_63/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_63/RestoreV2	RestoreV2save_63/Constsave_63/RestoreV2/tensor_names"save_63/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_63/AssignAssignlayer/W/Weightssave_63/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_63/Assign_1Assignlayer/b/biasessave_63/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_63/restore_allNoOp^save_63/Assign^save_63/Assign_1
;
save_64/ConstConst*
dtype0*
valueB Bmodel
g
save_64/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_64/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_64/SaveV2SaveV2save_64/Constsave_64/SaveV2/tensor_namessave_64/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_64/control_dependencyIdentitysave_64/Const^save_64/SaveV2*
T0* 
_class
loc:@save_64/Const
y
save_64/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_64/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_64/RestoreV2	RestoreV2save_64/Constsave_64/RestoreV2/tensor_names"save_64/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_64/AssignAssignlayer/W/Weightssave_64/RestoreV2*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(*
T0
�
save_64/Assign_1Assignlayer/b/biasessave_64/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_64/restore_allNoOp^save_64/Assign^save_64/Assign_1
;
save_65/ConstConst*
valueB Bmodel*
dtype0
g
save_65/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_65/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_65/SaveV2SaveV2save_65/Constsave_65/SaveV2/tensor_namessave_65/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_65/control_dependencyIdentitysave_65/Const^save_65/SaveV2*
T0* 
_class
loc:@save_65/Const
y
save_65/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_65/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_65/RestoreV2	RestoreV2save_65/Constsave_65/RestoreV2/tensor_names"save_65/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_65/AssignAssignlayer/W/Weightssave_65/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_65/Assign_1Assignlayer/b/biasessave_65/RestoreV2:1*
validate_shape(*
use_locking(*
T0*!
_class
loc:@layer/b/biases
?
save_65/restore_allNoOp^save_65/Assign^save_65/Assign_1
;
save_66/ConstConst*
valueB Bmodel*
dtype0
g
save_66/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_66/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_66/SaveV2SaveV2save_66/Constsave_66/SaveV2/tensor_namessave_66/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_66/control_dependencyIdentitysave_66/Const^save_66/SaveV2*
T0* 
_class
loc:@save_66/Const
y
save_66/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_66/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_66/RestoreV2	RestoreV2save_66/Constsave_66/RestoreV2/tensor_names"save_66/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_66/AssignAssignlayer/W/Weightssave_66/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_66/Assign_1Assignlayer/b/biasessave_66/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_66/restore_allNoOp^save_66/Assign^save_66/Assign_1
;
save_67/ConstConst*
valueB Bmodel*
dtype0
g
save_67/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_67/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
�
save_67/SaveV2SaveV2save_67/Constsave_67/SaveV2/tensor_namessave_67/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_67/control_dependencyIdentitysave_67/Const^save_67/SaveV2*
T0* 
_class
loc:@save_67/Const
y
save_67/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_67/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_67/RestoreV2	RestoreV2save_67/Constsave_67/RestoreV2/tensor_names"save_67/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_67/AssignAssignlayer/W/Weightssave_67/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_67/Assign_1Assignlayer/b/biasessave_67/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_67/restore_allNoOp^save_67/Assign^save_67/Assign_1
;
save_68/ConstConst*
valueB Bmodel*
dtype0
g
save_68/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_68/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_68/SaveV2SaveV2save_68/Constsave_68/SaveV2/tensor_namessave_68/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_68/control_dependencyIdentitysave_68/Const^save_68/SaveV2*
T0* 
_class
loc:@save_68/Const
y
save_68/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_68/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_68/RestoreV2	RestoreV2save_68/Constsave_68/RestoreV2/tensor_names"save_68/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_68/AssignAssignlayer/W/Weightssave_68/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_68/Assign_1Assignlayer/b/biasessave_68/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_68/restore_allNoOp^save_68/Assign^save_68/Assign_1
;
save_69/ConstConst*
valueB Bmodel*
dtype0
g
save_69/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_69/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_69/SaveV2SaveV2save_69/Constsave_69/SaveV2/tensor_namessave_69/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_69/control_dependencyIdentitysave_69/Const^save_69/SaveV2* 
_class
loc:@save_69/Const*
T0
y
save_69/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_69/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_69/RestoreV2	RestoreV2save_69/Constsave_69/RestoreV2/tensor_names"save_69/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_69/AssignAssignlayer/W/Weightssave_69/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_69/Assign_1Assignlayer/b/biasessave_69/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_69/restore_allNoOp^save_69/Assign^save_69/Assign_1
;
save_70/ConstConst*
valueB Bmodel*
dtype0
g
save_70/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_70/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_70/SaveV2SaveV2save_70/Constsave_70/SaveV2/tensor_namessave_70/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_70/control_dependencyIdentitysave_70/Const^save_70/SaveV2*
T0* 
_class
loc:@save_70/Const
y
save_70/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_70/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_70/RestoreV2	RestoreV2save_70/Constsave_70/RestoreV2/tensor_names"save_70/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_70/AssignAssignlayer/W/Weightssave_70/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_70/Assign_1Assignlayer/b/biasessave_70/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_70/restore_allNoOp^save_70/Assign^save_70/Assign_1
;
save_71/ConstConst*
valueB Bmodel*
dtype0
g
save_71/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_71/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_71/SaveV2SaveV2save_71/Constsave_71/SaveV2/tensor_namessave_71/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_71/control_dependencyIdentitysave_71/Const^save_71/SaveV2*
T0* 
_class
loc:@save_71/Const
y
save_71/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_71/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_71/RestoreV2	RestoreV2save_71/Constsave_71/RestoreV2/tensor_names"save_71/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_71/AssignAssignlayer/W/Weightssave_71/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_71/Assign_1Assignlayer/b/biasessave_71/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_71/restore_allNoOp^save_71/Assign^save_71/Assign_1
;
save_72/ConstConst*
valueB Bmodel*
dtype0
g
save_72/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_72/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_72/SaveV2SaveV2save_72/Constsave_72/SaveV2/tensor_namessave_72/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_72/control_dependencyIdentitysave_72/Const^save_72/SaveV2*
T0* 
_class
loc:@save_72/Const
y
save_72/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_72/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_72/RestoreV2	RestoreV2save_72/Constsave_72/RestoreV2/tensor_names"save_72/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_72/AssignAssignlayer/W/Weightssave_72/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_72/Assign_1Assignlayer/b/biasessave_72/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_72/restore_allNoOp^save_72/Assign^save_72/Assign_1
;
save_73/ConstConst*
valueB Bmodel*
dtype0
g
save_73/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_73/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_73/SaveV2SaveV2save_73/Constsave_73/SaveV2/tensor_namessave_73/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_73/control_dependencyIdentitysave_73/Const^save_73/SaveV2*
T0* 
_class
loc:@save_73/Const
y
save_73/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_73/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_73/RestoreV2	RestoreV2save_73/Constsave_73/RestoreV2/tensor_names"save_73/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_73/AssignAssignlayer/W/Weightssave_73/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_73/Assign_1Assignlayer/b/biasessave_73/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_73/restore_allNoOp^save_73/Assign^save_73/Assign_1
;
save_74/ConstConst*
valueB Bmodel*
dtype0
g
save_74/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_74/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_74/SaveV2SaveV2save_74/Constsave_74/SaveV2/tensor_namessave_74/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_74/control_dependencyIdentitysave_74/Const^save_74/SaveV2*
T0* 
_class
loc:@save_74/Const
y
save_74/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_74/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_74/RestoreV2	RestoreV2save_74/Constsave_74/RestoreV2/tensor_names"save_74/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_74/AssignAssignlayer/W/Weightssave_74/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_74/Assign_1Assignlayer/b/biasessave_74/RestoreV2:1*
validate_shape(*
use_locking(*
T0*!
_class
loc:@layer/b/biases
?
save_74/restore_allNoOp^save_74/Assign^save_74/Assign_1
;
save_75/ConstConst*
valueB Bmodel*
dtype0
g
save_75/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_75/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_75/SaveV2SaveV2save_75/Constsave_75/SaveV2/tensor_namessave_75/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_75/control_dependencyIdentitysave_75/Const^save_75/SaveV2*
T0* 
_class
loc:@save_75/Const
y
save_75/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_75/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_75/RestoreV2	RestoreV2save_75/Constsave_75/RestoreV2/tensor_names"save_75/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_75/AssignAssignlayer/W/Weightssave_75/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_75/Assign_1Assignlayer/b/biasessave_75/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_75/restore_allNoOp^save_75/Assign^save_75/Assign_1
;
save_76/ConstConst*
dtype0*
valueB Bmodel
g
save_76/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_76/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_76/SaveV2SaveV2save_76/Constsave_76/SaveV2/tensor_namessave_76/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_76/control_dependencyIdentitysave_76/Const^save_76/SaveV2*
T0* 
_class
loc:@save_76/Const
y
save_76/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_76/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_76/RestoreV2	RestoreV2save_76/Constsave_76/RestoreV2/tensor_names"save_76/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_76/AssignAssignlayer/W/Weightssave_76/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_76/Assign_1Assignlayer/b/biasessave_76/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_76/restore_allNoOp^save_76/Assign^save_76/Assign_1
;
save_77/ConstConst*
dtype0*
valueB Bmodel
g
save_77/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_77/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_77/SaveV2SaveV2save_77/Constsave_77/SaveV2/tensor_namessave_77/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_77/control_dependencyIdentitysave_77/Const^save_77/SaveV2*
T0* 
_class
loc:@save_77/Const
y
save_77/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_77/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_77/RestoreV2	RestoreV2save_77/Constsave_77/RestoreV2/tensor_names"save_77/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_77/AssignAssignlayer/W/Weightssave_77/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_77/Assign_1Assignlayer/b/biasessave_77/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_77/restore_allNoOp^save_77/Assign^save_77/Assign_1
;
save_78/ConstConst*
valueB Bmodel*
dtype0
g
save_78/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_78/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_78/SaveV2SaveV2save_78/Constsave_78/SaveV2/tensor_namessave_78/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_78/control_dependencyIdentitysave_78/Const^save_78/SaveV2* 
_class
loc:@save_78/Const*
T0
y
save_78/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_78/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_78/RestoreV2	RestoreV2save_78/Constsave_78/RestoreV2/tensor_names"save_78/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_78/AssignAssignlayer/W/Weightssave_78/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_78/Assign_1Assignlayer/b/biasessave_78/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_78/restore_allNoOp^save_78/Assign^save_78/Assign_1
;
save_79/ConstConst*
valueB Bmodel*
dtype0
g
save_79/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_79/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
�
save_79/SaveV2SaveV2save_79/Constsave_79/SaveV2/tensor_namessave_79/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_79/control_dependencyIdentitysave_79/Const^save_79/SaveV2* 
_class
loc:@save_79/Const*
T0
y
save_79/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_79/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_79/RestoreV2	RestoreV2save_79/Constsave_79/RestoreV2/tensor_names"save_79/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_79/AssignAssignlayer/W/Weightssave_79/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_79/Assign_1Assignlayer/b/biasessave_79/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_79/restore_allNoOp^save_79/Assign^save_79/Assign_1
;
save_80/ConstConst*
valueB Bmodel*
dtype0
g
save_80/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_80/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
�
save_80/SaveV2SaveV2save_80/Constsave_80/SaveV2/tensor_namessave_80/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_80/control_dependencyIdentitysave_80/Const^save_80/SaveV2*
T0* 
_class
loc:@save_80/Const
y
save_80/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_80/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_80/RestoreV2	RestoreV2save_80/Constsave_80/RestoreV2/tensor_names"save_80/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_80/AssignAssignlayer/W/Weightssave_80/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_80/Assign_1Assignlayer/b/biasessave_80/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_80/restore_allNoOp^save_80/Assign^save_80/Assign_1
;
save_81/ConstConst*
valueB Bmodel*
dtype0
g
save_81/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_81/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_81/SaveV2SaveV2save_81/Constsave_81/SaveV2/tensor_namessave_81/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_81/control_dependencyIdentitysave_81/Const^save_81/SaveV2*
T0* 
_class
loc:@save_81/Const
y
save_81/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_81/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_81/RestoreV2	RestoreV2save_81/Constsave_81/RestoreV2/tensor_names"save_81/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_81/AssignAssignlayer/W/Weightssave_81/RestoreV2*
validate_shape(*
use_locking(*
T0*"
_class
loc:@layer/W/Weights
�
save_81/Assign_1Assignlayer/b/biasessave_81/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_81/restore_allNoOp^save_81/Assign^save_81/Assign_1
;
save_82/ConstConst*
valueB Bmodel*
dtype0
g
save_82/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_82/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_82/SaveV2SaveV2save_82/Constsave_82/SaveV2/tensor_namessave_82/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_82/control_dependencyIdentitysave_82/Const^save_82/SaveV2*
T0* 
_class
loc:@save_82/Const
y
save_82/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_82/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_82/RestoreV2	RestoreV2save_82/Constsave_82/RestoreV2/tensor_names"save_82/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_82/AssignAssignlayer/W/Weightssave_82/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_82/Assign_1Assignlayer/b/biasessave_82/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_82/restore_allNoOp^save_82/Assign^save_82/Assign_1
;
save_83/ConstConst*
valueB Bmodel*
dtype0
g
save_83/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_83/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
�
save_83/SaveV2SaveV2save_83/Constsave_83/SaveV2/tensor_namessave_83/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_83/control_dependencyIdentitysave_83/Const^save_83/SaveV2*
T0* 
_class
loc:@save_83/Const
y
save_83/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_83/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_83/RestoreV2	RestoreV2save_83/Constsave_83/RestoreV2/tensor_names"save_83/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_83/AssignAssignlayer/W/Weightssave_83/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_83/Assign_1Assignlayer/b/biasessave_83/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_83/restore_allNoOp^save_83/Assign^save_83/Assign_1
;
save_84/ConstConst*
valueB Bmodel*
dtype0
g
save_84/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_84/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
�
save_84/SaveV2SaveV2save_84/Constsave_84/SaveV2/tensor_namessave_84/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_84/control_dependencyIdentitysave_84/Const^save_84/SaveV2*
T0* 
_class
loc:@save_84/Const
y
save_84/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_84/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_84/RestoreV2	RestoreV2save_84/Constsave_84/RestoreV2/tensor_names"save_84/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_84/AssignAssignlayer/W/Weightssave_84/RestoreV2*
validate_shape(*
use_locking(*
T0*"
_class
loc:@layer/W/Weights
�
save_84/Assign_1Assignlayer/b/biasessave_84/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_84/restore_allNoOp^save_84/Assign^save_84/Assign_1
;
save_85/ConstConst*
valueB Bmodel*
dtype0
g
save_85/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_85/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_85/SaveV2SaveV2save_85/Constsave_85/SaveV2/tensor_namessave_85/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_85/control_dependencyIdentitysave_85/Const^save_85/SaveV2*
T0* 
_class
loc:@save_85/Const
y
save_85/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_85/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_85/RestoreV2	RestoreV2save_85/Constsave_85/RestoreV2/tensor_names"save_85/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_85/AssignAssignlayer/W/Weightssave_85/RestoreV2*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(*
T0
�
save_85/Assign_1Assignlayer/b/biasessave_85/RestoreV2:1*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(*
T0
?
save_85/restore_allNoOp^save_85/Assign^save_85/Assign_1
;
save_86/ConstConst*
valueB Bmodel*
dtype0
g
save_86/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_86/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_86/SaveV2SaveV2save_86/Constsave_86/SaveV2/tensor_namessave_86/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_86/control_dependencyIdentitysave_86/Const^save_86/SaveV2* 
_class
loc:@save_86/Const*
T0
y
save_86/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_86/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_86/RestoreV2	RestoreV2save_86/Constsave_86/RestoreV2/tensor_names"save_86/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_86/AssignAssignlayer/W/Weightssave_86/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_86/Assign_1Assignlayer/b/biasessave_86/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_86/restore_allNoOp^save_86/Assign^save_86/Assign_1
;
save_87/ConstConst*
valueB Bmodel*
dtype0
g
save_87/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_87/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
�
save_87/SaveV2SaveV2save_87/Constsave_87/SaveV2/tensor_namessave_87/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_87/control_dependencyIdentitysave_87/Const^save_87/SaveV2*
T0* 
_class
loc:@save_87/Const
y
save_87/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_87/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_87/RestoreV2	RestoreV2save_87/Constsave_87/RestoreV2/tensor_names"save_87/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_87/AssignAssignlayer/W/Weightssave_87/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_87/Assign_1Assignlayer/b/biasessave_87/RestoreV2:1*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(*
T0
?
save_87/restore_allNoOp^save_87/Assign^save_87/Assign_1
;
save_88/ConstConst*
valueB Bmodel*
dtype0
g
save_88/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_88/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
�
save_88/SaveV2SaveV2save_88/Constsave_88/SaveV2/tensor_namessave_88/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_88/control_dependencyIdentitysave_88/Const^save_88/SaveV2*
T0* 
_class
loc:@save_88/Const
y
save_88/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_88/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_88/RestoreV2	RestoreV2save_88/Constsave_88/RestoreV2/tensor_names"save_88/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_88/AssignAssignlayer/W/Weightssave_88/RestoreV2*
validate_shape(*
use_locking(*
T0*"
_class
loc:@layer/W/Weights
�
save_88/Assign_1Assignlayer/b/biasessave_88/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_88/restore_allNoOp^save_88/Assign^save_88/Assign_1
;
save_89/ConstConst*
valueB Bmodel*
dtype0
g
save_89/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_89/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_89/SaveV2SaveV2save_89/Constsave_89/SaveV2/tensor_namessave_89/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_89/control_dependencyIdentitysave_89/Const^save_89/SaveV2*
T0* 
_class
loc:@save_89/Const
y
save_89/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_89/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_89/RestoreV2	RestoreV2save_89/Constsave_89/RestoreV2/tensor_names"save_89/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_89/AssignAssignlayer/W/Weightssave_89/RestoreV2*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(*
use_locking(
�
save_89/Assign_1Assignlayer/b/biasessave_89/RestoreV2:1*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(*
T0
?
save_89/restore_allNoOp^save_89/Assign^save_89/Assign_1
;
save_90/ConstConst*
valueB Bmodel*
dtype0
g
save_90/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_90/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_90/SaveV2SaveV2save_90/Constsave_90/SaveV2/tensor_namessave_90/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_90/control_dependencyIdentitysave_90/Const^save_90/SaveV2*
T0* 
_class
loc:@save_90/Const
y
save_90/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_90/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_90/RestoreV2	RestoreV2save_90/Constsave_90/RestoreV2/tensor_names"save_90/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_90/AssignAssignlayer/W/Weightssave_90/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_90/Assign_1Assignlayer/b/biasessave_90/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_90/restore_allNoOp^save_90/Assign^save_90/Assign_1
;
save_91/ConstConst*
valueB Bmodel*
dtype0
g
save_91/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_91/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_91/SaveV2SaveV2save_91/Constsave_91/SaveV2/tensor_namessave_91/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_91/control_dependencyIdentitysave_91/Const^save_91/SaveV2*
T0* 
_class
loc:@save_91/Const
y
save_91/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_91/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_91/RestoreV2	RestoreV2save_91/Constsave_91/RestoreV2/tensor_names"save_91/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_91/AssignAssignlayer/W/Weightssave_91/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_91/Assign_1Assignlayer/b/biasessave_91/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_91/restore_allNoOp^save_91/Assign^save_91/Assign_1
;
save_92/ConstConst*
valueB Bmodel*
dtype0
g
save_92/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_92/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_92/SaveV2SaveV2save_92/Constsave_92/SaveV2/tensor_namessave_92/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_92/control_dependencyIdentitysave_92/Const^save_92/SaveV2*
T0* 
_class
loc:@save_92/Const
y
save_92/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_92/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_92/RestoreV2	RestoreV2save_92/Constsave_92/RestoreV2/tensor_names"save_92/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_92/AssignAssignlayer/W/Weightssave_92/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_92/Assign_1Assignlayer/b/biasessave_92/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_92/restore_allNoOp^save_92/Assign^save_92/Assign_1
;
save_93/ConstConst*
valueB Bmodel*
dtype0
g
save_93/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_93/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_93/SaveV2SaveV2save_93/Constsave_93/SaveV2/tensor_namessave_93/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_93/control_dependencyIdentitysave_93/Const^save_93/SaveV2*
T0* 
_class
loc:@save_93/Const
y
save_93/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_93/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_93/RestoreV2	RestoreV2save_93/Constsave_93/RestoreV2/tensor_names"save_93/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_93/AssignAssignlayer/W/Weightssave_93/RestoreV2*
validate_shape(*
use_locking(*
T0*"
_class
loc:@layer/W/Weights
�
save_93/Assign_1Assignlayer/b/biasessave_93/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_93/restore_allNoOp^save_93/Assign^save_93/Assign_1
;
save_94/ConstConst*
valueB Bmodel*
dtype0
g
save_94/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_94/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_94/SaveV2SaveV2save_94/Constsave_94/SaveV2/tensor_namessave_94/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_94/control_dependencyIdentitysave_94/Const^save_94/SaveV2*
T0* 
_class
loc:@save_94/Const
y
save_94/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_94/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_94/RestoreV2	RestoreV2save_94/Constsave_94/RestoreV2/tensor_names"save_94/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_94/AssignAssignlayer/W/Weightssave_94/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_94/Assign_1Assignlayer/b/biasessave_94/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_94/restore_allNoOp^save_94/Assign^save_94/Assign_1
;
save_95/ConstConst*
valueB Bmodel*
dtype0
g
save_95/SaveV2/tensor_namesConst*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
N
save_95/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
�
save_95/SaveV2SaveV2save_95/Constsave_95/SaveV2/tensor_namessave_95/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_95/control_dependencyIdentitysave_95/Const^save_95/SaveV2*
T0* 
_class
loc:@save_95/Const
y
save_95/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_95/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueBB B 
�
save_95/RestoreV2	RestoreV2save_95/Constsave_95/RestoreV2/tensor_names"save_95/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_95/AssignAssignlayer/W/Weightssave_95/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_95/Assign_1Assignlayer/b/biasessave_95/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_95/restore_allNoOp^save_95/Assign^save_95/Assign_1
;
save_96/ConstConst*
valueB Bmodel*
dtype0
g
save_96/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_96/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_96/SaveV2SaveV2save_96/Constsave_96/SaveV2/tensor_namessave_96/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_96/control_dependencyIdentitysave_96/Const^save_96/SaveV2*
T0* 
_class
loc:@save_96/Const
y
save_96/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_96/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_96/RestoreV2	RestoreV2save_96/Constsave_96/RestoreV2/tensor_names"save_96/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_96/AssignAssignlayer/W/Weightssave_96/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_96/Assign_1Assignlayer/b/biasessave_96/RestoreV2:1*
validate_shape(*
use_locking(*
T0*!
_class
loc:@layer/b/biases
?
save_96/restore_allNoOp^save_96/Assign^save_96/Assign_1
;
save_97/ConstConst*
valueB Bmodel*
dtype0
g
save_97/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_97/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B 
�
save_97/SaveV2SaveV2save_97/Constsave_97/SaveV2/tensor_namessave_97/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_97/control_dependencyIdentitysave_97/Const^save_97/SaveV2*
T0* 
_class
loc:@save_97/Const
y
save_97/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_97/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_97/RestoreV2	RestoreV2save_97/Constsave_97/RestoreV2/tensor_names"save_97/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_97/AssignAssignlayer/W/Weightssave_97/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer/W/Weights*
validate_shape(
�
save_97/Assign_1Assignlayer/b/biasessave_97/RestoreV2:1*
T0*!
_class
loc:@layer/b/biases*
validate_shape(*
use_locking(
?
save_97/restore_allNoOp^save_97/Assign^save_97/Assign_1
;
save_98/ConstConst*
valueB Bmodel*
dtype0
g
save_98/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_98/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_98/SaveV2SaveV2save_98/Constsave_98/SaveV2/tensor_namessave_98/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_98/control_dependencyIdentitysave_98/Const^save_98/SaveV2*
T0* 
_class
loc:@save_98/Const
y
save_98/RestoreV2/tensor_namesConst"/device:CPU:0*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
`
"save_98/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_98/RestoreV2	RestoreV2save_98/Constsave_98/RestoreV2/tensor_names"save_98/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_98/AssignAssignlayer/W/Weightssave_98/RestoreV2*
validate_shape(*
use_locking(*
T0*"
_class
loc:@layer/W/Weights
�
save_98/Assign_1Assignlayer/b/biasessave_98/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_98/restore_allNoOp^save_98/Assign^save_98/Assign_1
;
save_99/ConstConst*
valueB Bmodel*
dtype0
g
save_99/SaveV2/tensor_namesConst*4
value+B)Blayer/W/WeightsBlayer/b/biases*
dtype0
N
save_99/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0
�
save_99/SaveV2SaveV2save_99/Constsave_99/SaveV2/tensor_namessave_99/SaveV2/shape_and_sliceslayer/W/Weightslayer/b/biases*
dtypes
2
q
save_99/control_dependencyIdentitysave_99/Const^save_99/SaveV2*
T0* 
_class
loc:@save_99/Const
y
save_99/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*4
value+B)Blayer/W/WeightsBlayer/b/biases
`
"save_99/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0
�
save_99/RestoreV2	RestoreV2save_99/Constsave_99/RestoreV2/tensor_names"save_99/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_99/AssignAssignlayer/W/Weightssave_99/RestoreV2*
validate_shape(*
use_locking(*
T0*"
_class
loc:@layer/W/Weights
�
save_99/Assign_1Assignlayer/b/biasessave_99/RestoreV2:1*
use_locking(*
T0*!
_class
loc:@layer/b/biases*
validate_shape(
?
save_99/restore_allNoOp^save_99/Assign^save_99/Assign_1
:
ArgMax/dimensionConst*
value	B :*
dtype0
^
ArgMaxArgMaxlayer/final_resultArgMax/dimension*
output_type0	*

Tidx0*
T0
<
ArgMax_1/dimensionConst*
value	B :*
dtype0
]
ArgMax_1ArgMaxinput/y_inputArgMax_1/dimension*
T0*
output_type0	*

Tidx0
)
EqualEqualArgMaxArgMax_1*
T0	
;
CastCastEqual*

DstT0*

SrcT0
*
Truncate( 
3
ConstConst*
dtype0*
valueB: 
?
MeanMeanCastConst*

Tidx0*
	keep_dims( *
T0"