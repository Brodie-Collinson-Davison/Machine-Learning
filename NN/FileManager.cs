using System;
using System.IO;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;
using NN;

static public class FileManager
{
    static public void MakeDirectory (String name, String path)
    {
        String fp = System.IO.Path.Combine(path, name);
        Directory.CreateDirectory(fp);
    }

    // Return a NeuralNetwork object from a json file
    static public NeuralNetwork DeserializeJSON ( String fileName )
    {
        NeuralNetwork net;

        try
        {
            String jsonString = File.ReadAllText(fileName);
            net = JsonSerializer.Deserialize<NeuralNetwork>(jsonString);
        }
        catch (Exception e)
        {
            // ensure null value
            net = null;
            UI.Display_Error(e.Message);
        }

        return net;
    }

    // Save a NeuralNetwork object to a JSON file
    static public void SerializeAsJson ( Object obj, string fileName )
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = false
        };

        // use inbuilt .net json serializer
        string jsonString = JsonSerializer.Serialize(obj, options);
        WriteFile(fileName, jsonString);
    }

    static public FileStream GetFileStream (string fileName, FileMode fMode)
    {
        FileStream stream = null;

        try
        {
            stream = new FileStream(fileName, fMode);

        } catch (IOException e)
        {
            UI.Display_Error(e.Message, true);

            if (stream != null)
            {
                try
                {
                    stream.Close();
                } catch (Exception e2)
                {
                    UI.Display_Error(e2.Message, true);
                }
            }
        }

        return stream;
    }

    static public void WriteFile(string fileName, string text)
    {
        using (System.IO.StreamWriter file = new System.IO.StreamWriter(Path.Combine(Directory.GetCurrentDirectory(), fileName)))
        {
            file.WriteLine(text);
            file.Close();
        }
    }

    static public void WriteFile(string fileName, string[] text)
    {
        using (System.IO.StreamWriter file = new System.IO.StreamWriter(Directory.GetCurrentDirectory() + "/" + fileName))
        {
            for (int i = 0; i < text.Length; i++)
            {
                file.WriteLine(text[i]);
            }

            file.Close();
        }
    }

    static public void AppendToFile(string fileName, string text)
    {
        using (System.IO.StreamWriter file = new System.IO.StreamWriter(Directory.GetCurrentDirectory() + "/" + fileName, true))
        {
            file.WriteLine(text);
            file.Close();
        }
    }

    static public void AppendToFile(string fileName, string[] text)
    {
        using (System.IO.StreamWriter file = new System.IO.StreamWriter(Directory.GetCurrentDirectory() + "/" + fileName, true))
        {
            for (int i = 0; i < text.Length; i++)
            {
                file.WriteLine(text[i]);
            }

            file.Close();
        }
    }

    static public string ReadFile(string fileName)
    {
       string s = File.ReadAllText(fileName);
       return s;
    }
}


