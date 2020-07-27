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
  PrintMultiArray(T.Grad);
  AssertTrue(ArrayEqual(T.Grad, Ones([2, 3]) / 2));
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

