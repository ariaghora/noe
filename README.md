<div align="center">
<img src="assets/noe-txt.png" alt="logo" width="300px"></img>
</div>

Hi. This is noe. This used to be darkteal. Noe is designed to support n-dimensional array. Please check the `master` branch for darkteal's code.

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

// Reshape C into a 6x1 tensor
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