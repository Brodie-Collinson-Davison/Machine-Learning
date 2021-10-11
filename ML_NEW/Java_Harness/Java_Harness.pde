import java.util.*;
import org.json.JSONObject;

/*
      GLOBALS 
*/

String SAVE_FOLDER_NAME = "SAVED_NETWORKS";

// options
String OPTIONS_FILE_NAME = "Visualiser-Options.json";
String networkName = "";
int textSize = 10;

int brushSize = 12;
int sqrBrushSize;
int pixelSize;

NN net;

String predictionText = "";
PImage canvas;

/*
//
//    PROCESSING FUNCTIONS 
//
*/

void setup ()
{ 
  // read startup options and apply any configurations
  readOptionsFile ();
  applyOptions ();
  
  // initialise canvas with black background
  size ( 375, 375 );
  background ( 0 );
  
  // calculate value once to save processing later
  sqrBrushSize = (int)pow (brushSize, 2);
  
  // initialise the drawing canvas
  canvas = createImage (width,height,RGB);
  
  // normalise canvas 'pixels' to fit mnist data
  pixelSize = width/28;

  // load network
  try
  {
      String filePath = sketchPath ();
      filePath = filePath.concat ("/../");
      filePath = filePath.concat (SAVE_FOLDER_NAME);
      filePath = filePath.concat ("/");
      filePath = filePath.concat (networkName);
      
      net = new NN (FileIO.readJSONFileAsObject (filePath));
  }
  catch ( RuntimeException e )
  {
    println (e.getMessage ());
  }
  
  println ("Setup complete");
}

void draw ()
{
  background (0);
  
  if ( mousePressed )
  {
    drawCircleAroundCursor ( brushSize  ); 
  } 
  
  image (canvas, 0, 0);
  text (predictionText, 0, 25);
}

void mouseDragged ()
{  
    Matrix mat = getInputFromCanvas();
    Matrix prediction = net.predict (mat);
    int label = getPrediction (prediction);
    float confidence = prediction.values[label];
    
    // display prediction
    predictionText = "Prediction: " + label + "    Confidence: " + (int)(confidence * 100) + "%";
}

void keyPressed ()
{
  // clear the canvas
  canvas = createImage (width, height, RGB);
}

/*
      FUNCTIONS 
*/

void readOptionsFile ()
{ 
  String fp = sketchPath().concat ("/");
  fp = fp.concat (OPTIONS_FILE_NAME);
  
  JSONObject obj = FileIO.readJSONFileAsObject ( fp );
  networkName = obj.getString ("Network_Name");
  textSize = obj.getInt ("Text_Size");
  brushSize = obj.getInt ("Brush_Size");
}

void applyOptions ()
{
  textSize (textSize);
}

int getPrediction (Matrix prediciton)
{
  float highestOutput = 0;
  int label = 0;
  for ( int i = 0; i < 10; i ++ )
  {
    if (prediciton.get(i,0) > highestOutput)
    {
        highestOutput = prediciton.get(i,0);
        label = i;
    }
  }

  return label;
}

// euclidean distance from a pixel(x,y) to the mouse position(mX,mY)
float distFromCursor (int x, int y)
{
  float dx = mouseX - x;
  float dy = mouseY - y;
  return sqrt (pow(dx,2) + pow(dy,2));
}

// returns the square euclidean distance from pixel(x,y) to the mouse position(mX,mY)
// faster operation than the euclidean distance
float sqrDistFromCursor (int x, int y)
{
  float dx = mouseX - x;
  float dy = mouseY - y;
  return pow(dx,2) + pow(dy,2);
}

void drawCircleAroundCursor (int sqrRadius)
{
  int cx = mouseX;
  int cy = mouseY;

  canvas.loadPixels ();
  for ( int y = cy - sqrRadius; y < cy + sqrRadius; y ++ )
  {
    for ( int x = cx - sqrRadius; x < cx + sqrRadius; x ++ )
    {
      float r = (int)round(sqrDistFromCursor (x,y));      
      if (r <= sqrRadius)
      {
        if (x > 0 && x < width && y > 0 && y < height )
        {
          canvas.pixels [y*width + x] = color (255,255,255);
        }
      }
    }
  }
  canvas.updatePixels ();
}

// Converts the canvas pixels into a matrix for NN 
Matrix getInputFromCanvas ()
{
  // create a copy of canvas
  PImage img = canvas.get();
  
  // resize image to mnist format (28 x 28 pixels)
  img.resize (28, 28);

  // load pixels into buffer
  img.loadPixels ();

  // convert pixel colours to greyscale values
  float[] data = new float [28*28];
  for ( int i = 0; i < img.pixels.length; i ++ )
  {
    int c=img.pixels[i];     // so we don't access the array too much
    int r=(c&0x00FF0000)>>16; // red part
    int g=(c&0x0000FF00)>>8; // green part
    int b=(c&0x000000FF);   // blue part
    int grey=(r+b+g)/3; 
    float val = grey/255f;
    data[i] = val;
  }
  
  // return as a column matrix
  return new Matrix (28*28, 1, data);
}
