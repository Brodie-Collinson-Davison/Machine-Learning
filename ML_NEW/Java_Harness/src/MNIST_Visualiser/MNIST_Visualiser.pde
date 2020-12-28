
import java.util.*;

int [] labels;
List<int [][]> images;

int label;
int[][] img;
int idx;

int pixelSize;

int getNewIndex () throws RuntimeException
{
   if ( labels.length == 0 )
   {
      throw new RuntimeException ( "Data incorrect" ); 
   }
   
   return (int)random ( labels.length );
}

void setup ()
{ 
  size ( 420, 420 );
  background ( 255 );
  noFill ();
  
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
}

void draw ()
{
  for ( int y = 0; y < 28; y ++ )
  {
     for ( int x = 0; x < 28; x ++ )
     {
        fill ( img [y][x] );
        stroke ( 0 );
        rect ( x*pixelSize, y*pixelSize, pixelSize, pixelSize );
     }
     println ("");
  }
}

void keyPressed ()
{
   idx = getNewIndex ();
   img = images.get ( idx );
}
