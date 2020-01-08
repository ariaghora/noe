<div align="center">
<img src="assets/noe-txt.png" alt="logo" width="300px"></img>
</div>

# Noe
Hi, mortal. This is noe.

```delphi
uses
  noe.core

var
  A, B, C: TTensor; 
```

And here we go.
```delphi
// Create a 2x3 tensor filled with 1
A := FullTensor([2, 3], 1);
PrintTensor(A);

// Create a 2x2x3 tensor filled with randomized values
B := FullTensor([2, 2, 3]);
PrintTensor(B);

// Create a 2x3 tensor with values specified
C := FullTensor([2, 3], [1, 2, 3, 4, 5, 6]);
PrintTensor(C);

// Reshape C into a 6x1 tensor
C.Reshape([6, 1]);
PrintTensor(C);
```