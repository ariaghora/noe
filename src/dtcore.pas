unit DTCore;

{$mode delphi}{$H+}

interface

uses
  Classes, SysUtils, dynlibs;

const
  {$IFDEF MSWINDOWS}
  BLAS_FILENAME = 'openblas.dll';
  {$ELSE}
  BLAS_FILENAME = 'openblas.so'; // not implemented yet
  {$ENDIF}

type
  {***** Primitive type wrapper *****}
  TFloatVector = array of double;

  {***** Callback function type wrapper *****}
  TCallbackDouble = function(x: double): double;

  {***** Darkteal-specific definitions *****}
  TDTMatrix = record
    val: TFloatVector;
    Width: longint;
    Height: longint;
    class operator Implicit(A: TFloatVector): TDTMatrix;
    class operator Explicit(A: TFloatVector): TDTMatrix;
    class operator Add(A, B: TDTMatrix): TDTMatrix;
    class operator Subtract(A, B: TDTMatrix): TDTMatrix; overload;
    class operator Subtract(A: TDTMatrix; x: double): TDTMatrix; overload;
    class operator Multiply(A: TDTMatrix; x: double): TDTMatrix; overload;
    class operator Multiply(x: double; A: TDTMatrix): TDTMatrix; overload;
    class operator Multiply(A, B: TDTMatrix): TDTMatrix; overload;
    class operator Divide(A: TDTMatrix; x: double): TDTMatrix; overload;
    class operator Divide(A, B: TDTMatrix): TDTMatrix; overload;
    function T: TDTMatrix;
    function GetRow(idx: integer): TDTMatrix;
    function GetColumn(idx: integer): TDTMatrix;
    function GetRange(row, col, Height, Width: longint): TDTMatrix;
    function Dot(A: TDTMatrix): TDTMatrix;
    function Apply(func: TCallbackDouble): TDTMatrix;
    function Sum(axis: integer = -1): TDTMatrix;
  end;


  {***** CBLAS-specific definition *****}
  CBLAS_ORDER = (CblasRowMajor = 101, CblasColMajor = 102);
  CBLAS_TRANSPOSE = (CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113);
  CBLAS_UPLO = (CblasUpper = 121, CblasLower = 122);
  CBLAS_DIAG = (CblasNonUnit = 131, CblasUnit = 132);

  _dcopy = procedure(N: longint; X: TFloatVector; incX: longint;
    Y: TFloatVector; incY: longint); cdecl;
  _daxpy = procedure(N: longint; alpha: double; X: TFloatVector;
    incX: longint; Y: TFloatVector; incY: longint); cdecl;
  _dscal = procedure(N: longint; alpha: double; X: TFloatVector;
    incX: longint); cdecl;
  _dgemm = procedure(Order: CBLAS_ORDER; TransA: CBLAS_TRANSPOSE;
    TransB: CBLAS_TRANSPOSE; M: longint; N: longint; K: longint;
    alpha: double; A: TFloatVector; lda: longint; B: TFloatVector;
    ldb: longint; beta: double; C: TFloatVector; ldc: longint); cdecl;
  _dtbmv = procedure(order: CBLAS_ORDER; Uplo: CBLAS_UPLO;
    TransA: CBLAS_TRANSPOSE; Diag: CBLAS_DIAG; N: longint; K: longint;
    A: TFloatVector; lda: longint; X: TFloatVector; incX: longint); cdecl;
  _ddot = function(N: longint; X: TFloatVector; incX: longint;
    Y: TFloatVector; incY: longint): double; cdecl;
  _dasum = function(N: longint; X: TFloatVector; incX: longint): double; cdecl;




procedure DarkTealInit;
procedure DarkTealRelease;

procedure PrintMatrix(M: TDTMatrix);
function CreateVector(size: integer; x: double): TFloatVector;
function CreateMatrix(row, col: integer; x: double): TDTMatrix; overload;
function CreateMatrix(row, col: integer): TDTMatrix; overload;
function Ones(row, col: integer): TDTMatrix;
function CopyMatrix(M: TDTMatrix): TDTMatrix;

function GetColumn(A: TDTMatrix; idx: integer): TDTMatrix;
function GetRow(A: TDTMatrix; idx: integer): TDTMatrix;

function Dot(A, B: TDTMatrix): TDTMatrix;
function Add(A, B: TDTMatrix): TDTMatrix;
function Subtract(A, B: TDTMatrix): TDTMatrix; overload;
function Subtract(A: TDTMatrix; x: double): TDTMatrix; overload;
function Multiply(A: TDTMatrix; x: double): TDTMatrix; overload;
function Multiply(A, B: TDTMatrix): TDTMatrix; overload;
function Divide(A: TDTMatrix; x: double): TDTMatrix; overload;
function Divide(A, B: TDTMatrix): TDTMatrix; overload;
function Sum(A: TDTMatrix): TDTMatrix; overload;
function Sum(A: TDTMatrix; axis: integer): TDTMatrix; overload;
function Max(A: TDTMatrix): double; overload;
function Max(A: TDTMatrix; axis: integer): TDTMatrix; overload;
function Min(A: TDTMatrix): double; overload;
function Min(A: TDTMatrix; axis: integer): TDTMatrix; overload;

function TileDown(A: TDTMatrix; size: integer): TDTMatrix; overload;

function Apply(func: TCallbackDouble; A: TDTMatrix): TDTMatrix;


function TDTMatrixFromCSV(f: string): TDTMatrix;

{$IFDEF FPC}
{$PACKRECORDS C}
{$ENDIF}

var
  blas_dcopy: _dcopy;
  blas_daxpy: _daxpy;
  blas_ddot: _ddot;
  blas_dscal: _dscal;
  blas_dgemm: _dgemm;
  blas_dtbmv: _dtbmv;
  blas_dasum: _dasum;
  libHandle: TLibHandle;

implementation

procedure DarkTealInit;
begin
  Assert(FileExists(BLAS_FILENAME), BLAS_FILENAME + ' cannot be found.');
  libHandle := LoadLibrary(BLAS_FILENAME);
  Pointer(@blas_dcopy) := GetProcedureAddress(libHandle, 'cblas_dcopy');
  Pointer(@blas_daxpy) := GetProcedureAddress(libHandle, 'cblas_daxpy');
  Pointer(@blas_ddot) := GetProcedureAddress(libHandle, 'cblas_ddot');
  Pointer(@blas_dscal) := GetProcedureAddress(libHandle, 'cblas_dscal');
  Pointer(@blas_dgemm) := GetProcedureAddress(libHandle, 'cblas_dgemm');
  Pointer(@blas_dtbmv) := GetProcedureAddress(libHandle, 'cblas_dtbmv');
  Pointer(@blas_dasum) := GetProcedureAddress(libHandle, 'cblas_dasum');
end;

procedure DarkTealRelease;
begin
  UnloadLibrary(libHandle);
  blas_dcopy := nil;
  blas_daxpy := nil;
  blas_ddot := nil;
  blas_dscal := nil;
  blas_dgemm := nil;
  blas_dtbmv := nil;
  blas_dasum := nil;
end;

class operator TDTMatrix.Explicit(A: TFloatVector): TDTMatrix;
begin
  Result.val := A;
end;

class operator TDTMatrix.Implicit(A: TFloatVector): TDTMatrix;
begin
  Result.val := A;
end;

class operator TDTMatrix.Add(A, B: TDTMatrix): TDTMatrix;
begin
  Result := DTCore.Add(A, B);
end;

class operator TDTMatrix.Subtract(A, B: TDTMatrix): TDTMatrix;
begin
  Result := DTCore.Subtract(A, B);
end;

class operator TDTMatrix.Subtract(A: TDTMatrix; x: double): TDTMatrix;
begin
  Result := DTCore.Subtract(A, x);
end;

class operator TDTMatrix.Multiply(A: TDTMatrix; x: double): TDTMatrix;
begin
  Result := DTCore.Multiply(A, x);
end;

class operator TDTMatrix.Multiply(x: double; A: TDTMatrix): TDTMatrix;
begin
  Result := DTCore.Multiply(A, x);
end;

class operator TDTMatrix.Multiply(A, B: TDTMatrix): TDTMatrix;
begin
  Result := DTCore.Multiply(A, B);
end;

class operator TDTMatrix.Divide(A: TDTMatrix; x: double): TDTMatrix;
begin
  Result := DTCore.Divide(A, x);
end;

class operator TDTMatrix.Divide(A, B: TDTMatrix): TDTMatrix;
begin
  Result := DTCore.Divide(A, B);
end;

function TDTMatrix.T: TDTMatrix;
var
  i, j, idx: longint;
begin
  Result := CopyMatrix(self);
  Result.Width := self.Height;
  Result.Height := self.Width;
  idx := 0;


  for i := 0 to self.Width - 1 do
    for j := 0 to self.Height - 1 do
    begin
      Result.val[idx] := self.val[j * self.Width + i];
      Inc(idx);
    end;

end;

function TDTMatrix.GetColumn(idx: integer): TDTMatrix;
begin
  Result := DTCore.GetColumn(self, idx);
end;

function TDTMatrix.GetRow(idx: integer): TDTMatrix;
begin
  Result := DTCore.GetRow(self, idx);
end;

function TDTMatrix.GetRange(row, col, Height, Width: longint): TDTMatrix;
var
  i, j, idx: integer;
begin
  Result.Width := Width;
  Result.Height := Height;
  SetLength(Result.val, Width * Height);
  idx := 0;
  for i := row to row + Height - 1 do
    for j := col to col + Width - 1 do
    begin
      Result.val[idx] := self.val[(i) * self.Width + (j)];
      Inc(idx);
    end;
end;

function TDTMatrix.Dot(A: TDTMatrix): TDTMatrix;
begin
  Result := DTCore.Dot(Self, A);
end;

function TDTMatrix.Apply(func: TCallbackDouble): TDTMatrix;
var
  i: longint;
begin
  Result := DTCore.CopyMatrix(self);
  for i := 0 to Length(Result.val) - 1 do
    Result.val[i] := func(self.val[i]);
end;

function TDTMatrix.Sum(axis: integer = -1): TDTMatrix;
begin
  Result := DTCore.Sum(self, axis);
end;

procedure PrintMatrix(M: TDTMatrix);
var
  i, j: integer;
begin
  for i := 0 to M.Height - 1 do
  begin
    Write('  ');
    for j := 0 to M.Width - 1 do
    begin
      Write(M.val[i * M.Width + j]: 5: 3, ' ');
    end;
    WriteLn;
  end;
  WriteLn;
end;

function CreateVector(size: integer; x: double): TFloatVector;
var
  i: integer;
  res: TFloatVector;
begin
  SetLength(res, size);
  for i := 0 to size - 1 do
    res[i] := x;
  Result := res;
end;

function CreateMatrix(row, col: integer; x: double): TDTMatrix;
var
  i: longint;
begin
  Result.Width := col;
  Result.Height := row;
  SetLength(Result.val, row * col);
  for i := 0 to (row * col) - 1 do
    Result.val[i] := x;
end;

function CreateMatrix(row, col: integer): TDTMatrix;
var
  i: longint;
begin
  Result.Width := col;
  Result.Height := row;
  SetLength(Result.val, row * col);
  for i := 0 to (row * col) - 1 do
    Result.val[i] := Random();
end;

function Ones(row, col: integer): TDTMatrix;
begin
  Result := createMatrix(row, col, 1);
end;

function CopyMatrix(M: TDTMatrix): TDTMatrix;
begin
  SetLength(Result.val, M.Width * M.Height);
  Result.Width := M.Width;
  Result.Height := M.Height;
  blas_dcopy(M.Width * M.Height, M.val, 1, Result.val, 1);
end;

function Dot(A, B: TDTMatrix): TDTMatrix;
begin
  SetLength(Result.val, A.Height * B.Width);
  Result.Height := A.Height;
  Result.Width := B.Width;
  blas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    A.Height, B.Width, B.Height, // m, n, k
    1, // alpha
    A.val, B.Height,
    B.val, B.Width,
    1, // beta
    Result.val, B.Width
    );
end;

function Multiply(A: TDTMatrix; x: double): TDTMatrix;
begin
  Result := DTCore.CopyMatrix(A);
  blas_dscal(A.Height * A.Width, x, Result.val, 1);
end;

function Multiply(A, B: TDTMatrix): TDTMatrix;
begin
  Result := CopyMatrix(B);
  blas_dtbmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
    Length(A.val), 0, A.val, 1, Result.val, 1);
end;

function Divide(A: TDTMatrix; x: double): TDTMatrix;
begin
  Result := DTCore.CopyMatrix(A);
  blas_dscal(A.Height * A.Width, 1 / x, Result.val, 1);
end;

function Divide(A, B: TDTMatrix): TDTMatrix;
var
  i: integer;
begin
  Result.Width := A.Width;
  Result.Height := A.Height;
  SetLength(Result.val, length(A.val));
  for i := 0 to length(Result.val) - 1 do
    Result.val[i] := A.val[i] / B.val[i];
end;

function GetColumn(A: TDTMatrix; idx: integer): TDTMatrix;
var
  i: integer;
begin
  SetLength(Result.val, A.Height);
  Result.Height := A.Height;
  Result.Width := 1;
  for i := 0 to Length(Result.val) - 1 do
    Result.val[i] := A.val[i * A.Width + idx];
end;

function GetRow(A: TDTMatrix; idx: integer): TDTMatrix;
var
  i: integer;
begin
  SetLength(Result.val, A.Width);
  Result.Height := 1;
  Result.Width := A.Width;
  for i := 0 to Length(Result.val) - 1 do
    Result.val[i] := A.val[idx * A.Width + i];
end;

function Sum(A: TDTMatrix): TDTMatrix;
begin
  Result.Width := 1;
  Result.Height := 1;
  SetLength(Result.val, 1);
  Result.val[0] := blas_dasum(A.Width * A.Height, A.val, 1);
end;

function Sum(A: TDTMatrix; axis: integer): TDTMatrix;
var
  i: integer;
begin
  if axis = 0 then
  begin
    SetLength(Result.val, A.Width);
    Result.Height := 1;
    Result.Width := A.Width;
    for i := 0 to A.Width - 1 do
      Result.val[i] := DTCore.Sum(GetColumn(A, i)).val[0];
  end
  else if axis = 1 then
  begin
    SetLength(Result.val, A.Height);
    Result.Height := A.Height;
    Result.Width := 1;
    for i := 0 to A.Height - 1 do
      Result.val[i] := DTCore.Sum(GetRow(A, i)).val[0];
  end
  else
  begin
    Result := sum(A);
  end;
end;

function Max(A: TDTMatrix): double;
var
  i: integer;
  CurMax: double;
begin
  CurMax := -1.0 / 0.0;
  for i := 0 to Length(A.val) - 1 do
    if A.val[i] > CurMax then
      CurMax := A.val[i];
  Result := CurMax;
end;

function Max(A: TDTMatrix; axis: integer): TDTMatrix;
var
  i: integer;
begin
  if axis = 0 then
  begin
    SetLength(Result.val, A.Width);
    Result.Height := 1;
    Result.Width := A.Width;
    for i := 0 to A.Width - 1 do
      Result.val[i] := DTCore.Max(GetColumn(A, i));
  end
  else
  begin
    SetLength(Result.val, A.Height);
    Result.Height := A.Height;
    Result.Width := 1;
    for i := 0 to A.Height - 1 do
      Result.val[i] := DTCore.Max(GetRow(A, i));
  end;
end;

function Min(A: TDTMatrix): double;
var
  i: integer;
  CurMin: double;
begin
  CurMin := 1.0 / 0.0;
  for i := 0 to Length(A.val) - 1 do
    if A.val[i] < CurMin then
      CurMin := A.val[i];
  Result := CurMin;
end;

function Min(A: TDTMatrix; axis: integer): TDTMatrix;
var
  i: integer;
begin
  if axis = 0 then
  begin
    SetLength(Result.val, A.Width);
    Result.Height := 1;
    Result.Width := A.Width;
    for i := 0 to A.Width - 1 do
      Result.val[i] := DTCore.Min(GetColumn(A, i));
  end
  else
  begin
    SetLength(Result.val, A.Height);
    Result.Height := A.Height;
    Result.Width := 1;
    for i := 0 to A.Height - 1 do
      Result.val[i] := DTCore.Min(GetRow(A, i));
  end;
end;

function TileDown(A: TDTMatrix; size: integer): TDTMatrix; overload;
var
  i, j: integer;
begin
  assert(A.Height = 1, 'Only matrix with height equals to 1 can be tiled down');
  Result.Width := A.Width;
  Result.Height := size;
  SetLength(Result.val, A.Width * A.Height * size);
  for i := 0 to size - 1 do
  begin
    for j := 0 to A.Width - 1 do
      Result.val[i * Result.Width + j] := A.val[j];
  end;
end;

function Apply(func: TCallbackDouble; A: TDTMatrix): TDTMatrix;
var
  i: longint;
begin
  Result := DTCore.CopyMatrix(A);
  for i := 0 to Length(Result.val) - 1 do
    Result.val[i] := func(A.val[i]);
end;

function Add(A, B: TDTMatrix): TDTMatrix;
begin
  Result := CopyMatrix(B);
  blas_daxpy(Length(A.val), 1, A.val, 1, Result.val, 1);
end;

function Subtract(A, B: TDTMatrix): TDTMatrix;
begin
  Result := CopyMatrix(A);
  blas_daxpy(Length(B.val), -1, B.val, 1, Result.val, 1);
end;

function Subtract(A: TDTMatrix; x: double): TDTMatrix;
begin
  Result := Subtract(A, CreateMatrix(A.Height, A.Width, x));
end;

function TDTMatrixFromCSV(f: string): TDTMatrix;
var
  tfIn: TextFile;
  s, row: string;
  i, idx, cntRow, cntCol: integer;
  isRowSet: boolean = False;
begin
  assignfile(tfIn, f);
  try
    Reset(tfIn); // open file
    cntRow := 0;
    idx := 0;
    s := '';
    SetLength(Result.val, 1);
    while not EOF(tfIn) do
    begin
      Readln(tfIn, row);
      if not isRowSet then
      begin
        //cntCol := Length(row);
        //isRowSet := True;
      end;

      Inc(cntRow);
      cntCol := 0;
      for i := 1 to Length(row) do
      begin
        if (row[i] <> ',') then
          s := s + row[i]
        else
        begin
          Inc(cntCol);
          Result.val[idx] := StrToFloat(s);

          s := '';
          Inc(idx);
          setLength(Result.val, idx + 1);
        end;
        if i = Length(row) then
        begin
          Inc(cntCol);
          Result.val[idx] := StrToFloat(s);
          Inc(idx);
          setLength(Result.val, idx + 1);
        end;
      end;
      s := '';
    end;
    CloseFile(tfIn); // close file
    //writeln(Result.val[0]);
  except
    on E: EInOutError do
      writeln('File handling error occurred. Details: ', E.Message);
  end;
  Result.Width := cntCol;
  Result.Height := cntRow;
  //Result := res;
end;

end.
