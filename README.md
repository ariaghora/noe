<div align="center">
<img src="assets/noe-txt.png" alt="logo" width="200px"></img>
</div>

This is my experimental project to understand the mechanism behind n-dimensional array: how's its layout in memory, how to do indexing, etc. Furthermore, I am also learning what happens under the hood of automatic differentiation for a neural network, so this project might be the good basis for it.

> This project used to be "darkteal". Please check the `master` branch for darkteal's code.

## Declaring and initializing tensors
```delphi
uses
  noe.core

var
  A, B, C: TTensor; 
```

A tensor filled with a specific value:
```delphi
{ 2x3 tensor filled with 1 }
A := FullTensor([2, 3], 1);
PrintTensor(A);
```

```
[[1.00, 1.00, 1.00]
 [1.00, 1.00, 1.00]]
```
A tensor of random values:
```delphi
{ 2x2x3 tensor filled with randomized values }
B := FullTensor([2, 2, 3]);
PrintTensor(B);
```

```
[[[0.83, 0.03, 0.96]
  [0.68, 0.91, 0.04]]

 [[0.45, 0.83, 0.70]
  [0.56, 0.31, 0.77]]]
```
Accessing the value of a tensor using a multidimensional index:
```delphi
{ Get the value of tensor A at index 0,1 }
WriteLn(A.GetVal([0, 1])); // will give 1

{ Get the value of tensor B at index 1,0,2 }
WriteLn(B.GetVal([1, 0, 1])); // will give 0.83
```
A tensor with specified values:
```delphi
{ 2x3 tensor filled with values specified }
C := FullTensor(
  [2, 3],      //--> target shape

  [1., 2., 3., //--> the data
   4., 5., 6.] //
);
PrintTensor(C);

WriteLn;

{ Reshape C into a 6x1 tensor }
C.Reshape([6, 1]);
PrintTensor(C);
```

```
[[1.00, 2.00, 3.00]
 [4.00, 5.00, 6.00]]

[[1.00]
 [2.00]
 [3.00]
 [4.00]
 [5.00]
 [6.00]]
```
Some basic element-wise arithmetical operations are also supported:
```delphi
A := FullTensor([3, 3], 1);
B := FullTensor([3, 3]);
WriteLn('A:');
PrintTensor(A);
WriteLn('B:');
PrintTensor(B);

WriteLn('A + B:');
PrintTensor(A + B);

WriteLn('A - B:');
PrintTensor(A - B);

WriteLn('A * B:');
PrintTensor(A * B);
```

## Note

- There is no broadcasting mechanism yet. For arithmetical operations, make sure the dimension of the operands matches. Broadcasting is also in my learning list. So, it is to be implemented.
- `GetValue` can only access a single value. So, the dimension of the index should matches the dimension of the tensor.
- Performance is not of my primary consideration, at least for now. I want simply a quick proof of concept of what I am learning.