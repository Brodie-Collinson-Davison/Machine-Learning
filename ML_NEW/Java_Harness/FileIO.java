/*
File:       FileIO.java
Author:     Brodie Collinson Davison
LastMod:    7/11/20
References:     [1] Leary, Sean (UserName: stleary). 2020. stleary/JSON-java. Github.com. https://github.com/stleary/JSON-java
*/

import java.util.*;
import java.io.*;

// (Leary. 2020)
import org.json.*;

/** Class:          FileIO
 * Purpose:
 * This class handles reading and writing to files.
 * Included are functions used to parse .json files into JSONObjects
 */
public class FileIO
{

    /*
    Name:           readFile
    Imports:        fileName (String) - name of file to read
    Exports:        lines (String []) - array containing lines of the file
    Purpose:        Reads a file with given filename located in the current directory.
                    Returns an array containing every line in the file.
    */
    public static String[] readFile ( String fileName )
    {
        String [] lines = null;
        FileInputStream fstrm = null;
        InputStreamReader rdr = null;
        BufferedReader bufRdr = null;

        List<String> linesList = new ArrayList<String>();
        Iterator<String> itr = null;
        int counter = 0;

        try
        {   
            // open file and setup input stream
            fstrm = new FileInputStream ( fileName );
            rdr = new InputStreamReader ( fstrm );
            bufRdr = new BufferedReader ( rdr );

            String currentLine = bufRdr.readLine ();

            while ( currentLine != null )
            {
                // store current line
                linesList.add (currentLine);

                // get next line
                currentLine = bufRdr.readLine ();
            }

            // close file
            fstrm.close ();

            // generate lines array
            lines = new String [linesList.size()];

            // iterate through lines list and insert
            itr = linesList.iterator ();
            
            while ( itr.hasNext () )
            {
                lines [counter] = (String)itr.next ();
                counter ++;
            }

        }
        catch ( IOException e )
        {   
            if ( fstrm != null )
            {
                // file is still open, attempt to close again
                try 
                {
                    fstrm.close ();
                } catch ( IOException e2 ) { }
            }

            String errorMessage = "Failed to open / read file " + fileName + "\n";
            errorMessage += e.getMessage ();

            System.out.println ( errorMessage );
        }

        return lines;
    }

    /*
    Name:           readJSONFileAsObject 
    Imports:        fileName (String) - name of .json file (include the .json extension)
    Exports:        jsonObj (JSONObject) - object contained by the file
    Purpose:        Parse a .json file and read everything into a json object
    Reference:      (Leary. 2020)
    */
    public static JSONObject readJSONFileAsObject ( String fileName )
    {
        JSONObject jsonObj = null;
        String fString;

        try
        {
            // read .json file as plain text
            String [] lines = readFile ( fileName );

            // append lines together into one string
            fString = "";
            for ( int i = 0; i < lines.length; i ++ )
            {
                fString += lines [i];
            }

            // construct object from string
            jsonObj = new JSONObject ( fString );
        }
        catch ( JSONException e )
        {
            String errMsg = "Failed to parse .json file: " + fileName + " " + e.getMessage ();
            System.out.println ( errMsg );
        }

        return jsonObj;
    }//readJSONFileAsObject()

    /*
    Name:           readJSONFileAsArray 
    Imports:        fileName (String) - name of .json file (include the .json extension)
    Exports:        array (JSONArray) - array of JSONObjects
    Purpose:        Parses a .json file and returns the JSON array stored inside.
    Assertion:      * provided file must only contain an array in a json format
    Reference:      (Leary. 2020)
    */
    public static JSONArray readJSONFileAsArray ( String fileName )
    {
        JSONArray array = null;
        String fString;

        try
        {
            // read .json file as plain text
            String [] lines = readFile ( fileName );

            // append lines together into one string
            fString = "";
            for ( int i = 0; i < lines.length; i ++ )
            {
                fString += lines [i];
            }

            // construct a json array out of the string
            array = new JSONArray ( fString );
        }
        catch ( JSONException e )
        {
            String errMsg = "Failed to parse .json file: " + fileName + " " + e.getMessage ();
            System.out.print (errMsg);
        }

        return array;
    }//readJSONFileAsArray()

}//FileIO{}
