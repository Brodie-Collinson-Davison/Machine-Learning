using System;

static public class UI
{
    //
    //      DISPLAY FUNCTIONS
    //


    // Displays menu and handles user interaction
    // Returns the selected menu option (indexed from 1)
    static public int MenuTree ( String menuName, String[] menuOptions, String additionalText = "" )
    {
        bool flag_exit = false;
        int menu_state = 1;

        // menu interaction loop
        while (!flag_exit)
        {
            // display menu title in the centre of screen
            SkpLn(2);
            DisplayCentered(menuName);
            SkpLn(2);
            
            // display additional dialogue
            if (!additionalText.Equals(""))
            {
                Console.WriteLine (additionalText);
                SkpLn(2);
            }

            // display menu options
            DisplayMenuTree (menu_state, menuOptions);

            // read user input
            ConsoleKey input = Console.ReadKey ().Key;
            
            // menu motion with arrow keys
            if ( input == ConsoleKey.UpArrow && menu_state > 1)
            {
                menu_state --;
            }
            else if ( input == ConsoleKey.DownArrow && menu_state < menuOptions.Length)
            {
                menu_state ++;
            }

            // clear display
            Console.Clear();

            // check for option selection
            if (input == ConsoleKey.Enter )
            {
                flag_exit = true;
            }
            
            // check for direct option selection
            // ascii numbers are from 48 to 57
            if ( (int)input > 47 && (int)input < 58 )
            {
                int num = (int)input - 48;

                if ( num > 0 && num <= menuOptions.Length )
                {
                    menu_state = num;
                    flag_exit = true;
                }
            }
        }
        return menu_state;
    }

    // Displays a menu with given options 
    // State reffers to the currently selected option
    static void DisplayMenuTree ( int state, String[] menuOptions )
    {
        for ( int i = 1; i <= menuOptions.Length; i ++ )
        {
            
            if ( state == i )
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write ("-");
                Console.Write ("[{0}]\t", i);
                Console.ForegroundColor = ConsoleColor.White;
            }
            else
            {
                Console.Write ( "{0})\t", i);  
            }
            Console.WriteLine (menuOptions[i - 1]);

            SkpLn ();
        }
    }

    // Displays a bold error to the user
    // assertContinue will halt the program until the user acknowledges
    static public void Display_Error (String message, bool assertContinue = false)
    {
        // display message
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine (message);
        Console.ResetColor ();

        // halt and wait for user acknowledge
        if (assertContinue)
            AssertContinue();
    }

    // Skip lines
    static public void SkpLn ( int numSkips = 1 )
    {
        for ( int i = 0; i < numSkips; i ++ )
        {
            Console.WriteLine ("");
        }
    }

    /// <summary>
    /// Displays a progress bar that will by default span the width of the window
    /// </summary>
    /// <param name="percent"> (decimal) percentage completion of progress bar </param>
    /// <param name="barSize"> size (chars) of the progress bar </param>
    static public void DisplayProgressBar ( float percent, int barSize = 0 )
    {
        if (barSize == 0)
            barSize = Console.WindowWidth - 2;

        Console.Write("[");
        for (int i = 0; i < barSize; i++)
        {
            if ((float)i / (float)barSize < percent )
                Console.Write("#");
            else
                Console.Write(" ");
        }
        Console.WriteLine("]\n");
    }
    

    /// <summary>
    /// Displays text in the centre of the console window
    /// </summary>
    /// <param name="text"> text to display </param>
    static public void DisplayCentered (String text)
    {
        Console.CursorLeft = (Console.WindowWidth / 2) - (text.Length / 2);
        Console.WriteLine(text);
    }

    //
    //      INTERACTIVE METHODS
    //

    // Halt the program until the user acknowledges the message
    static public void AssertContinue()
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.Write("Press enter to continue:");
        Console.ReadLine();
        Console.ResetColor();
    }

    //Handles user interaction for number input
    //Returns inputted number
    static public float Query_Float(String msg, float min = float.MinValue, float max = float.MaxValue)
    {
        bool flag_exit = false;
        float output = 0;

        while (!flag_exit)
        {
            // display input query instructions
            Console.WriteLine(msg);

            // display bounds
            if (min != float.MinValue && max != float.MaxValue)
            {
                Console.WriteLine("Input Bounds: {0} > input > {1}", min, max);
            }
            else if (min != float.MinValue)
            {
                Console.WriteLine("Input Bounds: {0} > input", min);
            }
            else if (max != float.MaxValue)
            {
                Console.WriteLine("Input Bounds: input < {0", max);
            }

            String inputStr;

            try
            {
                inputStr = Console.ReadLine();
                output = float.Parse(inputStr);

                if (output >= min && output <= max)
                    flag_exit = true;
                else
                    throw new ArgumentOutOfRangeException("input was not within bounds!");
            }
            catch (Exception e)
            {
                inputStr = "";
                UI.Display_Error(e.Message, false);
            }
        }

        return output;
    }

    //Handles user interaction for number input
    //Returns inputted number
    static public int Query_Int (String msg, int min = int.MinValue, int max = int.MaxValue)
    {
        bool flag_exit = false;
        int output = 0;

        while (!flag_exit)
        {
            // display input query instructions
            Console.WriteLine(msg);

            // display bounds
            if (min != int.MinValue && max != int.MaxValue)
            {
                Console.WriteLine("Input Bounds: {0} > input > {1}", min, max);
            }
            else if (min != int.MinValue)
            {
                Console.WriteLine("Input Bounds: {0} > input", min);
            }
            else if (max != int.MaxValue)
            {
                Console.WriteLine("Input Bounds: input < {0", max);
            }

            String inputStr;

            try
            {
                inputStr = Console.ReadLine();
                output = int.Parse(inputStr);

                if (output >= min && output <= max)
                    flag_exit = true;
                else
                    throw new ArgumentOutOfRangeException("input was not within bounds!");
            }
            catch (Exception e)
            {
                inputStr = "";
                UI.Display_Error(e.Message, false);
            }
        }

        return output;
    }


    /// <summary>
    /// Handles user interaction to ask for a yes / no answer
    /// </summary>
    /// <param name="start_state"> default condition of the answer </param>
    /// <param name="sure_check">  </param>
    /// <returns></returns>
    static public bool YesNoQuestion ( bool start_state = false, bool sure_check = false )
    {
        SkpLn ( 2 );

        bool completedQuestion = false;
        bool state = start_state;


        while (!completedQuestion)
        {
            // print current state of the question
            if (state)
                Console.WriteLine ("( [Y] , n )? ");
            else
                Console.WriteLine ("( y , [N] )? ");

            // get input
            ConsoleKeyInfo input = Console.ReadKey ();
            
            // change descision with arrow key
            if ( input.Key == ConsoleKey.LeftArrow || 
                 input.Key == ConsoleKey.RightArrow )
            {
                Console.SetCursorPosition (0, Console.CursorTop - 1);
                state = !state;
            }

            // answer yes check
            if ( input.Key == ConsoleKey.Y )
            {
                state = true;
                completedQuestion = true;
            } 

            // answer no check
            if ( input.Key == ConsoleKey.N )
            {
                state = false;
                completedQuestion = false;
            }    

            // answer current selection check
            if ( input.Key == ConsoleKey.Enter )
                completedQuestion = true;
        }

        // recursively ask question to double check if user is sure of their action
        if ( sure_check )
        {
            Console.WriteLine("Are you sure?");
            state = YesNoQuestion(start_state);
        }

        return state;
    }

}