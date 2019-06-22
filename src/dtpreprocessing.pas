{ A unit containing data preprocessing functionality. }
unit DTPreprocessing;

{$mode delphi}

interface

uses
  Classes, SysUtils, DTCore, DTUtils, fgl;

type
  TMap = TFPGMap<double, longint>;

  { @abstract(A class to encode labels in a one-vs-all fashion.) }
  TOneHotEncoder = class
    uniqueLabels: TFloatVector;
    mapperEncode: TMap;
  private
    function CreateOneHotVector(idx: integer): TFloatVector;
  public
    function Fit(y: TDTMatrix): TOneHotEncoder;
    function Transform(y: TDTMatrix): TDTMatrix;
  end;

  { @abstract(A class to scale the data between 0 and 1.)
    The min and max are calculated per-column, by which each column is scaled
    between 0 and 1. }
  TMinMaxScaler = class
    Maxs: TDTMatrix;
    Mins: TDTMatrix;
  private
  public
    function Fit(X: TDTMatrix): TMinMaxScaler;
    function Transform(X: TDTMatrix): TDTMatrix;
  end;

implementation

function TOneHotEncoder.CreateOneHotVector(idx: integer): TFloatVector;
var
  res: TFloatVector;
begin
  res := CreateVector(Length(self.uniqueLabels), 0);
  res[self.mapperEncode.Data[idx]] := 1;
  Result := res;
end;

function TOneHotEncoder.Fit(y: TDTMatrix): TOneHotEncoder;
var
  i: longint;
begin
  Assert(y.Width = 1, 'The shape of y must be 1 by n.');
  self.uniqueLabels := GetUnique(y.val);

  self.mapperEncode := TMap.Create;
  for i := 0 to Length(self.uniqueLabels) - 1 do
  begin
    self.mapperEncode.Add(round(self.uniqueLabels[i]), i);
  end;

  Result := self;
end;

function TOneHotEncoder.Transform(y: TDTMatrix): TDTMatrix;
var
  i, j, idx: longint;
  row: TFloatVector;
begin
  idx := 0;
  Result.Width := length(self.uniqueLabels);
  Result.Height := y.Height;
  SetLength(Result.val, Result.Width * Result.Height);

  for i := 0 to y.Height - 1 do
  begin
    row := CreateOneHotVector(self.mapperEncode.KeyData[round(y.val[i])]);
    for j := 0 to Length(self.uniqueLabels) - 1 do
    begin
      Result.val[idx] := row[j];
      Inc(idx);
    end;
  end;
end;

function TMinMaxScaler.Fit(X: TDTMatrix): TMinMaxScaler;
var
  i: integer;
  Mins, Maxs: TDTMatrix;
begin
  self.Maxs := Max(X, 0);
  self.Mins := Min(X, 0);
  Result := self;
end;

function TMinMaxScaler.Transform(X: TDTMatrix): TDTMatrix;
begin
  Result := (X - TileDown(self.Mins, X.Height)) /
    (TileDown(self.Maxs, X.Height) - TileDown(self.Mins, X.Height));
end;

end.
