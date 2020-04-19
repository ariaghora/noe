unit noe2;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fgl, noe.utils, Math, noe.ndarr, noe.types;

type
  TTensor      = class;

  TTensor = class
  private
    Data: TNdArr;
    fRequiresGrad: boolean;
    function GetNDims: longint;
    function GetShape: TIntVector;
  public
    constructor Create;
    procedure Cleanup;
    property RequiresGrad: boolean read fRequiresGrad write fRequiresGrad;
    property Shape: TIntVector read GetShape;
    property NDims: longint read GetNDims;
  end;

  TTensorList = specialize TFPGList<TTensor>;


  procedure Cleanup;
  procedure PrintTensor2D(T: TTensor);

  function CreateEmptyTensor(Shape: array of longint): TTensor;
  function CreateTensor(Data: TNdArr): TTensor;
  function CreateTensor(Shape: array of longint; v: NFloat): TTensor;

  function Add(A, B: TTensor): TTensor;

  operator +(A, B: TTensor) C: TTensor;

implementation

uses
  noe.mathwrapper;

var
  tensorList: TTensorList;



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

function CreateEmptyTensor(Shape: array of longint): TTensor;
begin
  Result := TTensor.Create();
  Result.Data := CreateEmptyNdArr(Shape);
end;

function CreateTensor(Data: TNdArr): TTensor;
begin
  Result := CreateEmptyTensor(Data.Shape);
  Result.Data := Data;
end;

function CreateTensor(Shape: array of longint; v: NFloat): TTensor;
begin
  Result := CreateEmptyTensor(Shape);
  Result.Data.Fill(v);
end;


function Add(A, B: TTensor): TTensor;
begin
  Result := CreateTensor(ApplyBfunc(A.Data, B.Data, @Add_F));
end;

operator+(A, B: TTensor)C: TTensor;
begin
  C := Add(A, B);
end;

{ TTensor }

function TTensor.GetShape: TIntVector;
begin
  Exit(Self.Data.Shape);
end;

constructor TTensor.Create;
begin
  inherited;
  tensorList.Add(Self);
end;

function TTensor.GetNDims: longint;
begin
  Exit(self.Data.NDims);
end;

procedure TTensor.Cleanup;
begin
  self.Data.Cleanup;
  FreeAndNil(self);
end;

{ TNdArr }



initialization

tensorList := TTensorList.Create;

end.

