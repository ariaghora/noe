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
  TFloatVector = array of double;

  {***** CBLAS-specific definition *****}
  CBLAS_ORDER = (CblasRowMajor = 101, CblasColMajor = 102);
  CBLAS_TRANSPOSE = (CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113);


  _ddot = function(N: longint; X: TFloatVector; incX: longint;
    Y: TFloatVector; incY: longint): double; cdecl;
  _dgemm = procedure(Order: CBLAS_ORDER; TransA: CBLAS_TRANSPOSE;
    TransB: CBLAS_TRANSPOSE; M: longint; N: longint; K: longint;
    alpha: double; A: TFloatVector; lda: longint; B: TFloatVector;
    ldb: longint; beta: double; C: TFloatVector; ldc: longint); cdecl;

  {***** Darkteal-specific definitions *****}
  TDTMatrix = record
    val: TFloatVector;
    Width: longint;
    Height: longint;
    class operator Implicit(mat: TFloatVector): TDTMatrix;
    class operator Explicit(mat: TFloatVector): TDTMatrix;
    function Dot(A: TDTMatrix): TDTMatrix;
  end;




procedure DarkTealInit;
procedure PrintMatrix(M: TDTMatrix);
function CreateMatrix(row, col: integer; x: double): TDTMatrix; overload;
function CreateMatrix(row, col: integer): TDTMatrix; overload;

function Dot(A, B: TDTMatrix): TDTMatrix;
function Add(A, B: TDTMatrix): TDTMatrix;

{$IFDEF FPC}
{$PACKRECORDS C}
{$ENDIF}

var
  blas_ddot: _ddot;
  blas_dgemm: _dgemm;
  libHandle: TLibHandle;

implementation

class operator TDTMatrix.Explicit(mat: TFloatVector): TDTMatrix;
begin
  Result.val := mat;
end;

class operator TDTMatrix.Implicit(mat: TFloatVector): TDTMatrix;
begin
  Result.val := mat;
end;

function TDTMatrix.Dot(A: TDTMatrix): TDTMatrix;
begin
  Result := DTCore.Dot(Self, A);
end;

procedure DarkTealInit;
begin
  Assert(FileExists(BLAS_FILENAME), BLAS_FILENAME + ' cannot be found.');
  libHandle := LoadLibrary(BLAS_FILENAME);
  Pointer(@blas_ddot) := GetProcedureAddress(libHandle, 'cblas_ddot');
  Pointer(@blas_dgemm) := GetProcedureAddress(libHandle, 'cblas_dgemm');
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

function Add(A, B: TDTMatrix): TDTMatrix;
var
  i: longint;
begin
  setlength(Result.val, length(A.val));
  Result.Width := A.Width;
  Result.Height := A.Height;
  for i := 0 to Length(A.val) - 1 do
    Result.val[i] := A.val[i] + B.val[i];
end;

end.

