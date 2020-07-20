unit noe2;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fgl, noe.utils, Math, noe.ndarr, noe.types;

type
  TTensor     = class;
  TTensorList = specialize TFPGList<TTensor>;

  { TTensor }

  TTensor = class
  private
    fDependencies: TTensorList;
    fRequiresGrad: boolean;
    function GetNDims: longint;
    function GetShape: TIntVector;
  public
    Data: TNdArr;
    constructor Create;
    function Reshape(NewShape: array of longint): TTensor;
    function T: TTensor;
    procedure AddDependency(Deps: array of TTensor);
    procedure Cleanup;
    property Dependencies: TTensorList read fDependencies write fDependencies;
    property RequiresGrad: boolean read fRequiresGrad write fRequiresGrad;
    property Shape: TIntVector read GetShape;
    property NDims: longint read GetNDims;
  end;

  procedure Cleanup;
  procedure PrintTensor2D(T: TTensor);

  function CreateEmptyTensor(Shape: array of longint): TTensor;
  function CreateTensor(Data: TNdArr): TTensor;
  function CreateTensor(Data: array of NFloat): TTensor;
  function CreateTensor(Shape: array of longint; v: NFloat): TTensor;

  function Add(A, B: TTensor): TTensor;

  operator +(A, B: TTensor) C: TTensor;

var
  tensorList: TTensorList;

implementation

uses
  noe.mathwrapper;

procedure Cleanup;
var
  t: TTensor;
begin
  for t in tensorList do
    t.Cleanup;
  FreeAndNil(tensorList);
end;

procedure PrintTensor2D(T: TTensor);
var
  i, j: integer;
  s: string;
begin
  Assert(T.Data.NDims = 2, 'Can only print a tensor with NDims = 2.');
  s := '';

  if T.NDims = 0 then
    s := s + FloatToStr(T.Data.Val[0])
  else if T.NDims = 1 then
    for i := 0 to T.Shape[0] - 1 do
      s := s + FloatToStr(T.Data.Val[i]);   // ENSURE CONTIGUOUS

  for i := 0 to T.Data.Shape[0] - 1 do
  begin
    for j := 0 to T.Data.Shape[1] - 1 do
    begin
      s := s + FloatToStr(T.Data.Val[i * T.Data.Shape[1] + j]);
      if j < T.Data.Shape[1] - 1 then s := s + ' ';
    end;
    s := s + sLineBreak;
  end;
  WriteLn(s);
end;

function CreateEmptyTensor(Shape: array of longint): TTensor;
begin
  Result := TTensor.Create();
  Result.Data := CreateEmptyNdArr(Shape);
  Result.Dependencies := TTensorList.Create;
end;

function CreateTensor(Data: TNdArr): TTensor;
begin
  Result := CreateEmptyTensor(Data.Shape);
  Result.Data := Data;
end;

function CreateTensor(Data: array of NFloat): TTensor;
var
  i: longint;
begin
  Result := CreateEmptyTensor([Length(Data)]);
  for i := 0 to Result.Data.Size - 1 do
    Result.Data.Val[i] := Data[i];
end;

function CreateTensor(Shape: array of longint; v: NFloat): TTensor;
begin
  Result := CreateEmptyTensor(Shape);
  Result.Data.Fill(v);
end;

function Add(A, B: TTensor): TTensor;
begin
  Result := CreateTensor(ApplyBfunc(A.Data, B.Data, @Add_F));
  Result.AddDependency([A, B]);
end;

operator+(A, B: TTensor)C: TTensor;
begin
  C := Add(A, B);
end;

function TTensor.GetShape: TIntVector;
begin
  Exit(Self.Data.Shape);
end;

constructor TTensor.Create;
begin
  inherited;
  tensorList.Add(Self);
end;

function TTensor.Reshape(NewShape: array of longint): TTensor;
begin
  Result := CreateTensor(Self.Data.Reshape(NewShape));
end;

function TTensor.T: TTensor;
begin
  Result := CreateTensor(Self.Data.T.Contiguous());
end;

procedure TTensor.AddDependency(Deps: array of TTensor);
var
  d: TTensor;
begin
  for d in Deps do
  begin
    self.RequiresGrad := self.RequiresGrad or d.RequiresGrad;
    if d.RequiresGrad then
      self.Dependencies.Add(d);
  end;
end;

function TTensor.GetNDims: longint;
begin
  Exit(self.Data.NDims);
end;

procedure TTensor.Cleanup;
begin
  self.Data.Cleanup;
  FreeAndNil(self.fDependencies);
  FreeAndNil(self);
end;


initialization

tensorList := TTensorList.Create;

end.

