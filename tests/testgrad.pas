unit testgrad;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, noe, multiarray, numerik;

type

  TTestGrad = class(TTestCase)
  published
    procedure TestGradMax1;
    procedure TestGradMax2;
    procedure TestGradMean;
    procedure TestGradMeanAxis;
    procedure TestGradSigmoid;
    { Regular sum over elements }
    procedure TestGradSum;
    { Sum with axis=0 }
    procedure TestGradSumAxis1;
    { Sum with axis=1 }
    procedure TestGradSumAxis2;
    { Sum with axis=1 and KeepDims=True }
    procedure TestGradSumAxis3;

  end;

var
  A: TMultiArray;
  T, U: TTensor;

implementation

procedure TTestGrad.TestGradMax1;
begin
  T := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  T.RequiresGrad := True;
  U := Max(T);
  U.Backward(Ones(U.Shape));
  AssertTrue(ArrayEqual(T.Grad,
                        TMultiArray([0, 0, 0,
                                     0, 0, 1]).Reshape([2, 3])
                        ));
end;

procedure TTestGrad.TestGradMax2;
begin
  T := TMultiArray([6, 6, 6, 6, 6, 6]).Reshape([2, 3]);
  T.RequiresGrad := True;
  U := Max(T);
  U.Backward(Ones(U.Shape));
  AssertTrue(ArrayEqual(T.Grad,
                        Ones([2, 3])
                        ));
end;

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
  AssertTrue(ArrayEqual(T.Grad,
                        TMultiArray([0.1966, 0.1050, 0.0452,
                                     0.0177, 0.0066, 0.0025]).Reshape([2, 3]),
                        1e-4));
end;

procedure TTestGrad.TestGradSum;
begin
  T := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  T.RequiresGrad := True;
  U := Sum(T);
  U.Backward(Ones(U.Shape));
  AssertTrue(ArrayEqual(T.Grad, Ones([2, 3])));
end;

procedure TTestGrad.TestGradSumAxis1;
begin
  T := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  T.RequiresGrad := True;
  U := Sum(T, 0);
  U.Backward(Ones(U.Shape));
  AssertTrue(ArrayEqual(T.Grad, Ones([2, 3])));
end;

procedure TTestGrad.TestGradSumAxis2;
begin
  T := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  T.RequiresGrad := True;
  U := Sum(T, 1);
  U.Backward(Ones(U.Shape));
  AssertTrue(ArrayEqual(T.Grad, Ones([2, 3])));
end;

procedure TTestGrad.TestGradSumAxis3;
begin
  T := TMultiArray([1, 2, 3, 4, 5, 6]).Reshape([2, 3]);
  T.RequiresGrad := True;
  U := Sum(T, 1, True);
  U.Backward(Ones(U.Shape));
  AssertTrue(ArrayEqual(T.Grad, Ones([2, 3])));
end;


initialization

  RegisterTest(TTestGrad);
end.

