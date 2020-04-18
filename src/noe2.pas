unit noe2;

{$mode objfpc}{$H+}{$modeSwitch advancedRecords}

interface

uses
  Classes, SysUtils;

type
  NFloat = double;

  TIntVector   = array of longint;
  TFloatVector = array of NFloat;
  TTensor      = class;

  { TTensor }

  { TNdArr }

  TNdArr = record
  private
    FShape: array of longint;
    FStrides: array of longint;
    function GetNDims: longint;
    function GetSize: longint;
  public
    Val:     TFloatVector;
    function Dot(Other: TTensor): TTensor;
    function DumpCSV(Sep: string = ','): string;
    function GetAt(Index: array of longint): TTensor;
    function GetShape: TIntVector;
    function Reshape(ShapeVals: array of longint): TTensor;
    function T: TTensor;
    function ToTensor(RequiresGrad: boolean = False): TTensor;
    procedure Fill(v: double);
    procedure Cleanup;
    procedure SetAt(Index: array of longint; x: double);
    procedure WriteToCSV(FileName: string);
    procedure ReshapeInplace(NewShape: array of longint);
    property NDims: longint read GetNDims;
    property Shape: TIntVector read FShape write FShape;
    property Size: longint read GetSize;
    property Strides: TIntVector read FStrides write FStrides;
  end;

  TTensor = class
    Data: TNdArr;
    procedure Cleanup;
  end;

  procedure PrintTensor2D(T: TTensor);

  function CreateEmptyNdArr(Shape: array of longint): TNdArr;
  function CreateEmptyTensor(Shape: array of longint): TTensor;
  function CreateTensor(Shape: array of longint; v: NFloat): TTensor;

  function ShapeToSize(Shape: array of longint): longint;
  function ShapeToStride(Shape: array of longint): TIntVector;

implementation

procedure PrintTensor2D(T: TTensor);
var
  i, j: integer;
  s: string;
begin
  Assert(T.Data.NDims = 2, 'PrintTensor2D can only print a 2-tensor.');
  s := '';
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

function CreateEmptyNdArr(Shape: array of longint): TNdArr;
var
  size: LongInt;
begin
  size := ShapeToSize(Shape);
  SetLength(Result.Val, size);
  Result.ReshapeInplace(Shape);
  Result.Strides := ShapeToStride(Shape);
end;

function CreateEmptyTensor(Shape: array of longint): TTensor;
begin
  Result := TTensor.Create();
  Result.Data := CreateEmptyNdArr(Shape);
end;

function CreateTensor(Shape: array of longint; v: NFloat): TTensor;
begin
  Result := CreateEmptyTensor(Shape);
  Result.Data.Fill(v);
end;

function ShapeToSize(Shape: array of longint): longint;
var
  i, size: longint;
begin
  size := 1;
  for i := 0 to Length(Shape) - 1 do
    size := size * shape[i];
  Result := size;
end;

function ShapeToStride(Shape: array of longint): TIntVector;
var
  k, j, prod: longint;
begin
  SetLength(Result, Length(Shape));

  for k := 0 to Length(Shape) - 1 do
  begin
    prod := 1;
    for j := k + 1 to Length(Shape) - 1 do
      prod := prod * Shape[j];
    Result[k] := prod;
  end;
end;

{ TTensor }

procedure TTensor.Cleanup;
begin
  self.Data.Cleanup;
  FreeAndNil(self);
end;

{ TNdArr }

function TNdArr.GetNDims: longint;
begin
  Exit(Length(Self.Shape));
end;

function TNdArr.GetSize: longint;
begin
  Exit(Length(self.Val));
end;

function TNdArr.Dot(Other: TTensor): TTensor;
begin

end;

function TNdArr.DumpCSV(Sep: string): string;
begin

end;

function TNdArr.GetAt(Index: array of longint): TTensor;
begin

end;

function TNdArr.GetShape: TIntVector;
begin

end;

function TNdArr.Reshape(ShapeVals: array of longint): TTensor;
begin

end;

function TNdArr.T: TTensor;
begin

end;

function TNdArr.ToTensor(RequiresGrad: boolean): TTensor;
begin

end;

procedure TNdArr.Fill(v: double);
var
  i: longint;
begin
  for i := 0 to Self.Size - 1 do
    self.Val[i] := v;
end;

procedure TNdArr.Cleanup;
begin
  self.val := nil;
  self.Shape := nil;
  self.Strides := nil;
end;

procedure TNdArr.SetAt(Index: array of longint; x: double);
begin

end;

procedure TNdArr.WriteToCSV(FileName: string);
begin

end;

procedure TNdArr.ReshapeInplace(NewShape: array of longint);
var
  i: longint;
begin
  SetLength(self.FShape, Length(NewShape));
  for i := 0 to Length(NewShape) - 1 do
    self.FShape[i] := NewShape[i];
  self.Strides := ShapeToStride(NewShape);
end;


end.

