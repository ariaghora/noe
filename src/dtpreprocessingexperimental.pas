unit DTPreprocessingExperimental;

{$mode delphi}
//{$M+}

interface

uses
  Classes, SysUtils, DTCore, DTUtils, fgl;

type
  TMap = TFPGMap<double, longint>;

  TOneHotEncoder = class
    uniqueLabels: TFloatVector;
    mapperEncode: TMap;
  private
    function CreateOneHotVector(idx: integer): TFloatVector;
  public
    function Fit(y: TDTMatrix): TOneHotEncoder;
    function Transform(y: TDTMatrix): TDTMatrix;
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
  self.uniqueLabels := Getunique(y.val);

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

end.
