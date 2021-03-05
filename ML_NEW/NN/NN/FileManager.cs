using System;
using System.IO;
using System.Collections.Generic;
using System.Text.Json;

static public class FileManager
{
    static public void SerializeAsJson ( Object obj, string fileName )
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true
        };
        string jsonString = JsonSerializer.Serialize(obj, options);
        WriteFile(fileName, jsonString);
    }

    static public void WriteFile(string fileName, string text)
    {
        using (System.IO.StreamWriter file = new System.IO.StreamWriter(Directory.GetCurrentDirectory() + "/" + fileName))
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


