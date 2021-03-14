
import java.util.*;

/*
      GLOBALS 
*/

int [] labels;
List<int [][]> images;

int label;
int[][] img;
int idx;

int brushSize = 20;
int sqrBrushSize;
int pixelSize;

NN net;

PImage canvas;

/*
      FUNCTIONS 
*/

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

int getNewIndex () throws RuntimeException
{
   if ( labels.length == 0 )
   {
      throw new RuntimeException ( "Data incorrect" ); 
   }
   
   return (int)random ( labels.length );
}

float distFromCursor (int x, int y)
{
 float dx = mouseX - x;
 float dy = mouseY - y;
 return sqrt (pow(dx,2) + pow(dy,2));
}

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
  
  loadPixels();
  
  for ( int y = cy - sqrRadius; y < cy + sqrRadius; y ++ )
  {
    for ( int x = cx - sqrRadius; x < cx + sqrRadius; x ++ )
    {
      float r = (int)round(sqrDistFromCursor (x,y));      
      if (r <= sqrRadius)
      {
        if (x > 0 && x < width && y > 0 && y < height )
        {
          pixels [y*width + x] = color (255,255,255);
        }
      }
    }
  }
  
  updatePixels ();
}

Matrix getInputFromCanvas ()
{
  PImage img = createImage (width, height, GRAY);
  
  loadPixels (); 
  for ( int i = 0; i < pixels.length; i ++ )
  {
     img.pixels [i] = pixels[i]; 
  }
  img.updatePixels ();
  img.resize (28,28);
  
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
  
  Matrix input = new Matrix (28*28, 1, data);
  return input;
}

/*
      PROCESSING FUNCTIONS 
*/

void setup ()
{ 
  size ( 450, 450 );
  background ( 0 );
  
  sqrBrushSize = (int)pow (brushSize, 2);
  
  canvas = new PImage (width,height);
  
  pixelSize = width/28;
  
  // get mnist data
  try
  {
    labels = Formatter.getLabels ();
    images = Formatter.getImages ();
    
    idx = getNewIndex ();
    
    label = labels [idx];
    img = images.get ( idx );
  }
  catch ( RuntimeException e )
  {
    println ( "Unable to load MNIST Data: " );
    println ( e.getMessage () );
    exit ();
  }

  // load network
  try
  {
      net = new NN (FileIO.readJSONFileAsObject ("G:/Projects/ML/Machine-Learning/ML_NEW/Java_Harness/NN_Big_2.json"));
  }
  catch ( RuntimeException e )
  {
     println (e.getMessage ());
  }
}

void draw ()
{
  if ( mousePressed )
  {
    drawCircleAroundCursor ( sqrBrushSize  );
  } 
}

void mouseReleased ()
{
  /* display random mnist img
  idx = getNewIndex ();
  img = images.get ( idx );
  
  for ( int i = 0; i < 28; i ++ )
  {
    String str = "";
    for ( int j = 0; j < 28; j ++ )
    {
        float f = (float)img[i][j]/255.0f;
        String s = "";
        if ( f < 0.5f )
          s = ".";
        else if ( f < 0.75f )
          s = "+";
        else 
          s = "#";
        s += " ";
        
        str += s;
    }
    println (str);
  }
  */
    Matrix mat = getInputFromCanvas();
    Matrix prediction = net.predict (mat);
    int label = getPrediction (prediction);
    float confidence = prediction.values[label];
    
    // display canvas
  for ( int i = 0; i < 28; i ++ )
  {
    String str = "";
    for ( int j = 0; j < 28; j ++ )
    {
        float f = mat.values[i*28+j];
        String s = "";
        if ( f < 0.5f )
          s = ".";
        else if ( f < 0.75f )
          s = "+";
        else 
          s = "#";
        s += " ";
        
        str += s;
    }
    println (str);
  }
    // display prediction
    println ("Prediction: " + label + "\tConfidence: " + confidence );
  
}

void keyPressed ()
{
  clear();
   /*
   idx = getNewIndex ();
   img = images.get ( idx );

   Matrix input = Formatter.formatInput (img);
   Matrix prediction = net.predict (input);
   
   int label = getPrediction (prediction);
   
   if ( label != labels[idx] )
     System.out.print ("X\t");
   System.out.println ("Prediction: " + label + " Actual: " + labels[idx]);
    */
}
