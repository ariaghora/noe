unit noe.mathwrapper;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, noe.types;

function Add_F(v1, v2: NFloat): NFloat;

implementation

function Add_F(v1, v2: NFloat): NFloat;
begin
  Result := v1 + v2;
end;

end.

