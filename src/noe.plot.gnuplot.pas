{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 This unit provides an interface to GNU Plot.
}
unit noe.plot.gnuplot;

{$mode objfpc}{$H+}

interface

uses
  Classes, noe, SysUtils;

var
  { Hold global count of created plots }
  GlobalPlotCount: integer;

type
  TPlotType = (ptBoxes, ptLines, ptPoints, ptHistogram, ptImage);

  { @abstract(A record containing plot style) }
  TPlotStyle = record
    LineType:  longint;
    LineColor: string;
    LineWidth: longint;
    PointType: longint;
    PointSize: longint;
  end;

  { @abstract(A class that holds information of data points to plot, including its style) }
  TPlot = class
    PlotStyle: TPlotStyle;
    Title:     string;
    PlotType:  TPlotType;
    OverrideDefaultStyle: boolean;
  public
    Values: TTensor;
    constructor Create;
    { Set the data points to plot
      @param(x only accepts TDTMatrix with size of 1 by m or m by 1) }
    procedure SetDataPoints(x: TTensor); overload;
    { Set the data points to plot (x axis against y axis) }
    procedure SetDataPoints(x, y: TTensor); overload;
    function GenerateScript: string;
  private
    FileName: string;
    procedure WriteDataStringTableToFile;
    procedure RemoveDataStringTableFile;
  end;

  { @abstract(A class that holds information of a single figure) }
  TFigure = class(TObject)
    Title:   string;
    XLabel:  string;
    YLabel:  string;
    Palette: string;
  public
    constructor Create;
    procedure AddPlot(Plot: TPlot);
    procedure Show;
  private
    PlotList: TList;
    function GenerateScript: string;
  end;

{ Initialize plotting functionality by passing gnuplot executable path }
procedure GNUPlotInit(GNUplotPath: string);
procedure ImageShow(img: TTensor; title: string = '');


implementation

var
  _GNUPlotInitialized: boolean = False;
  _GNUPlotPath: string;
  _GNUPlotTerminal: string;

procedure MatrixStringTableToFile(X: TTensor; fn: string);
var
  F: TextFile;
begin
  AssignFile(F, fn);
  try
    ReWrite(F);
    Write(F, X.DumpCSV(' '));
  finally
    CloseFile(F);
  end;
end;

function IsDTPlotReady: boolean;
begin
  Result := True;
  if not _GNUPlotInitialized then
  begin
    WriteLn('GNU Plot has not been configured properly.');
    Result := False;
  end;
end;

procedure GNUPlotInit(GNUplotPath: string);
begin
  _GNUPlotPath     := GNUplotPath;
  _GNUPlotTerminal := 'qt';
  //if FileExists(GNUplotPath) then
  _GNUPlotInitialized := True;
  //else
  //  WriteLn('GNU Plot executable is not found.');
end;

procedure ImageShow(img: TTensor; title: string);
var
  fig: TFigure;
  plot: TPlot;
begin
  Assert(img.NDims = 2, 'Currently only greyscale images are supported (NDims=2).');
  fig := TFigure.Create;
  fig.Title := title;
  fig.Palette := 'grey';

  plot := TPlot.Create;
  plot.SetDataPoints(VFlip(img));
  plot.PlotType := ptImage;

  fig.AddPlot(plot);
  fig.Show;
end;

constructor TPlot.Create;
begin
  OverrideDefaultStyle := False;
  PlotType := ptPoints; // 'histogram', 'lines', 'dots'
  Inc(GlobalPlotCount);
  FileName := Format('_DTPLOT_TMP_%d.tmp', [GlobalPlotCount]);

  { default style (for overriding) }
  PlotStyle.LineType  := 1;
  PlotStyle.LineColor := '#000000';
  PlotStyle.LineWidth := 2;
  PlotStyle.PointType := 7;
  PlotStyle.PointSize := 1;
end;

procedure TPlot.RemoveDataStringTableFile;
begin
  if FileExists(self.FileName) then
    DeleteFile(self.FileName);
end;

procedure TPlot.WriteDataStringTableToFile;
begin
  MatrixStringTableToFile(self.Values, self.FileName);
end;

function TPlot.GenerateScript: string;
var
  s, style, PlotTypeStr, Modifier: string;
begin
  case PlotType of
    ptLines: PlotTypeStr     := 'lines';
    ptPoints: PlotTypeStr    := 'points';
    ptHistogram: PlotTypeStr := 'histogram';
    ptBoxes: PlotTypeStr     := 'boxes';
    ptImage: PlotTypeStr     := 'image';
  end;

  if PlotType = ptImage then
    Modifier := 'matrix';

  if not OverrideDefaultStyle then
    style := ''
  else
    style := Format(' linetype %d linecolor ''%s'' linewidth %d pointtype %d pointsize %d',
      [PlotStyle.LineType, PlotStyle.LineColor, PlotStyle.LineWidth,
      PlotStyle.PointType, PlotStyle.PointSize]);
  s := Format('''%s'' %s title ''%s'' with %s%s',
    [FileName, Modifier, Title, PlotTypeStr, style]);
  Result := s;
end;

procedure TPlot.SetDataPoints(x: TTensor);
var
  x_: TTensor;
begin
  x_ := x;

  if self.PlotType <> ptImage then
    x_.ReshapeInplace([x_.Size, 1]);

  self.Values := x_;
end;

procedure TPlot.SetDataPoints(x, y: TTensor);
var
  x_, y_: TTensor;
begin
  //if ((x_.Shape[1] = 1) or (x_.Shape[0] = 1)) and ((y_.Shape[1] = 1) or (y_.Shape[0] = 1)) then
  //begin
  //  x_ := CopyTensor(x);
  //  y_ := CopyTensor(y);
  //  if x.Shape[1] > 1 then
  //    x_ := x_.T;
  //  if y.Shape[1] > 1 then
  //    y_ := y_.T;
  //  self.Values := AppendColumns(x, y);
  //end;
end;

constructor TFigure.Create;
begin
  self.PlotList := TList.Create;
  self.Palette  := 'rgbformulae 7,5,15';
end;

function TFigure.GenerateScript: string;
var
  s, script: string;
  i: integer;
begin
  s := '' + sLineBreak;
  s := s + 'set terminal %s title ''%s'';' + sLineBreak;
  s := s + 'set key right top;' + sLineBreak;
  s := s + 'set xlabel ''' + self.XLabel + ''';' + sLineBreak;
  s := s + 'set ylabel ''' + self.YLabel + ''';' + sLineBreak;
  s := s + 'set palette ' + self.Palette + ';' + sLineBreak;

  s := s + 'do for [i=1:64] {set style line i linewidth 2};' + sLineBreak;

  s := s + 'plot ';
  for i := 0 to PlotList.Count - 1 do
  begin
    s := s + TPlot(PlotList.items[i]).GenerateScript;
    if i < PlotList.Count - 1 then
      s := s + ',';
  end;
  s      := s + ';';
  script := Format(s, [_GNUPlotTerminal, Title]);
  Result := script;
end;

procedure TFigure.AddPlot(Plot: TPlot);
begin
  PlotList.Add(Plot);
end;

procedure TFigure.Show;
var
  i: integer;
begin
  { Generate temp files for each plot }
  for i := 0 to PlotList.Count - 1 do
    TPlot(PlotList.Items[i]).WriteDataStringTableToFile;

  if IsDTPlotReady then
    ExecuteProcess(Utf8ToAnsi(Format('%s --persist -e "%s" ',
      [_GNUPlotPath, self.GenerateScript])),
      '', []);

  { do cleanup (temp files removal) }
  for i := 0 to PlotList.Count - 1 do
    TPlot(PlotList.Items[i]).RemoveDataStringTableFile;

end;

initialization
  GNUPlotInit('gnuplot');
  GlobalPlotCount := 0;

end.
