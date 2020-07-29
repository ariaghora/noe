unit testgrad;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, noe2, multiarray, numerik;

type

  TTestGrad = class(TTestCase)
  published
    procedure TestGradMean;
    procedure TestGradMeanAxis;
    procedure TestGradSigmoid;
    procedure TestGradSum;
  end;

var
  A: TMultiArray;
  T, U: TTensor;

implementation

procedure TTestGrad.TestGradMean;
begin
  T := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  T.RequiresGrad := True;
  U := Mean(T);
  U.Backward(Ones(U.Shape));
  AssertTrue(ArrayEqual(T.Grad, Ones([2, 3]) / 6));
end;

procedure TTestGrad.TestGradMeanAxis;
begin
  T := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  T.RequiresGrad := True;
  U := Mean(T, 0);
  U.Backward(Ones(U.Shape));
  AssertTrue(ArrayEqual(T.Grad, Ones([2, 3]) / 2));
end;

procedure TTestGrad.TestGradSigmoid;
begin
  T := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  T.RequiresGrad := True;
  U := Sigmoid(T);
  U.Backward(Ones(U.Shape));
  PrintTensor(T.Grad);
  PrintTensor(TMultiArray([0.1966, 0.1050, 0.0452, 0.0177, 0.0066, 0.0025]).Reshape([2, 3]));
  AssertTrue(ArrayEqual(T.Grad,
       TMultiArray([0.1966, 0.1050, 0.0452, 0.0177, 0.0066, 0.0025]).Reshape([2, 3])));
end;

procedure TTestGrad.TestGradSum;
begin
  T := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  T.RequiresGrad := True;
  U := Sum(T);
  U.Backward(Ones(U.Shape));
  AssertTrue(ArrayEqual(T.Grad, Ones([2, 3])));
end;


initialization

  RegisterTest(TTestGrad);
end.

