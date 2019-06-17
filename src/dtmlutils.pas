unit DTMLUtils;

{ A unit containing machine-learning related classes anc functions }

{$mode delphi}

interface

uses
  Classes, SysUtils, Math, DTCore, DTLinAlg;

function Sigmoid(x: double): double; overload;
function Sigmoid(A: TDTMatrix): TDTMatrix; overload;
function SigmoidPrime(x: double): double; overload;
function SigmoidPrime(A: TDTMatrix): TDTMatrix; overload;
function Relu(x: double): double; overload;
function Relu(A: TDTMatrix): TDTMatrix; overload;
function ReluPrime(x: double): double; overload;
function ReluPrime(A: TDTMatrix): TDTMatrix; overload;
function AccuracyScore(yhat, y: TDTMatrix): double;

procedure TrainTestSplit(X, y: TDTMatrix; split: double; var XTrain: TDTMatrix;
  var XTest: TDTMatrix; var yTrain: TDTMatrix; var yTest: TDTMatrix;
  stratified: boolean);



implementation

{ @abstract(Sigmoid activation function) }
function Sigmoid(x: double): double;
begin
  Result := 1 / (1 + exp(-x));
end;

function Sigmoid(A: TDTMatrix): TDTMatrix;
var
  i: integer;
begin
  Result := Apply(@Sigmoid, A);
end;

function SigmoidPrime(x: double): double;
begin
  Result := Sigmoid(x) * (1 - Sigmoid(x));
end;

function SigmoidPrime(A: TDTMatrix): TDTMatrix;
begin
  Result := Apply(@SigmoidPrime, A);
end;

function Relu(x: double): double;
begin
  Result := x * integer(x >= 0);
end;

function Relu(A: TDTMatrix): TDTMatrix;
begin
  Result := Apply(@Relu, A);
end;

function ReluPrime(x: double): double;
begin
  Result := 1 * integer(x >= 0);
end;

function ReluPrime(A: TDTMatrix): TDTMatrix;
begin
  Result := Apply(@ReluPrime, A);
end;

function AccuracyScore(yhat, y: TDTMatrix): double;
var
  i, tot: integer;
begin
  for i := 0 to Length(y.val) - 1 do
    if y.val[i] = yhat.val[i] then
      tot := tot + 1;
  Result := tot / Length(y.val);
end;

procedure TrainTestSplit(X, y: TDTMatrix; split: double; var XTrain: TDTMatrix;
  var XTest: TDTMatrix; var yTrain: TDTMatrix; var yTest: TDTMatrix;
  stratified: boolean);
var
  TrainSize, index: integer;
  XCopy, XtrainTmp, row: TDTMatrix;
begin
  Xtrain := CreateMatrix(0, X.Width);
  TrainSize := round(split * X.Height);
  XCopy := CopyMatrix(X);
  while Xtrain.Height < TrainSize do
  begin
    index := Random(XCopy.Height);
    row := PopRow(XCopy, index);
    Xtrain := AppendRows(Xtrain, row);
  end;
  XTest := XCopy;
end;

end.
