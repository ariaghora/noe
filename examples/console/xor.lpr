program xor_example;

{$mode objfpc}{$H+}

uses
  multiarray, numerik, noe;

var
  X, y, yPred, W1, W2, b1, b2, Loss: TTensor;
  LearningRate: Single;
  i: integer;

begin
  Randomize;

  X := CreateMultiArray([0, 0,
                         0, 1,
                         1, 0,
                         1, 1]).Reshape([4, 2]);
  y := CreateMultiArray([0, 1, 1, 0]).Reshape([4, 1]);

  W1 := Random([2, 5]); // Input to hidden
  W2 := Random([5, 1]); // Hidden to output
  W1.RequiresGrad := True;
  W2.RequiresGrad := True;

  b1 := Zeros([5]);
  b2 := Zeros([1]);
  b1.RequiresGrad := True;
  b2.RequiresGrad := True;

  LearningRate := 0.01;
  for i := 0 to 2000 do
  begin
    yPred := (ReLu(X.Matmul(W1) + b1)).Matmul(W2) + b2;
    Loss := Mean(Sqr(yPred - y)); // MSE error

    W1.ZeroGrad;
    W2.ZeroGrad;
    b1.ZeroGrad;
    b2.ZeroGrad;

    Loss.Backward();

    W1.Data := W1.Data - LearningRate * W1.Grad;
    W2.Data := W2.Data - LearningRate * W2.Grad;
    b1.Data := b1.Data - LearningRate * b1.Grad;
    b2.Data := b2.Data - LearningRate * b2.Grad;

    if i mod 50 = 0 then
      WriteLn('Loss at iteration ', i, ': ', Loss.Data.Get(0) : 5 : 2);
  end;

  WriteLn('Prediction:');
  PrintTensor(YPred);

  Write('Press enter to exit'); ReadLn;
end.

