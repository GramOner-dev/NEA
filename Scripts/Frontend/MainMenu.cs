using System;
using System.Collections.Generic;
using System.Threading;

//------------------------------
//Rewrote most of this whole thing about 4 times
//------------------------------

interface IMenuNavigator
{
    void ShowMenu(int id);
    void GoBack();
    void ExitProgram();
}

class MenuManager : IMenuNavigator
{
    private readonly Dictionary<int, MenuScreen> menus = new();
    private MenuScreen currentMenu;

    public void AddMenu(MenuScreen menu) => menus[menu.GetScreenID] = menu;

    public void ShowMenu(int id)
    {
        if (!menus.TryGetValue(id, out currentMenu))
        {
            throw new InvalidOperationException($"Menu ID {id} not found");
            Console.ReadKey();
            return;
        }

        Console.Clear();
        currentMenu.StartMenu(this);
    }
}

class MenuScreen
{
    private int ScreenID { get; private set; }
    private readonly List<Button> buttons = new();
    private int currentButtonIndex = 0;

    public MenuScreen(int id) => ScreenID = id;

    public void AddButton(Button button) => buttons.Add(button);

    public void StartMenu(IMenuNavigator navigator)
    {
        ConsoleKeyInfo key;
        bool running = true;

        while (running)
        {
            DrawMenu();

            key = Console.ReadKey(true);
            switch (key.Key)
            {
                case ConsoleKey.UpArrow:
                    changeButtonIndex(1);
                    break;

                case ConsoleKey.DownArrow:
                    changeButtonIndex(-1);
                    break;

            }
        }
    }
    private void changeButtonIndex(int direction)
    {
        buttons[currentButtonIndex].changeIsSelectedStateTo(false);
        currentButtonIndex = (currentButtonIndex + direction + buttons.Count) % buttons.Count;
        buttons[currentButtonIndex].changeIsSelectedStateTo(true);
    }

    private void DrawMenu()
    {
        Console.Clear();
        for (int i = 0; i < buttons.Count; i++)
        {
            if (i == currentButtonIndex)
                Console.WriteLine($"> {buttons[i].ButtonName}");
            else
                Console.WriteLine($"  {buttons[i].ButtonName}");
        }
        Thread.Sleep(50);
    }

    public int GetScreenID() => ScreenID;
}

public enum ButtonType
{
    ActionButton,
    ExitButton,
    SaveButton,
    RedirectButton
}

public class ButtonFactory
{
    IMenuNavigator navigatorReference;
    public ButtonFactory(IMenuNavigator navigator)
    {
        navigatorReference = navigator;
    }
    public Button CreateButton(ButtonType type, string name)
    {
        return type switch
        {
            ButtonType.ActionButton => new ActionButton(name, navigatorReference),
            ButtonType.ExitButton => new ExitButton(name, navigatorReference),
            ButtonType.SaveButton => new SaveButton(name, navigatorReference),
            _ => throw new ArgumentOutOfRangeException(nameof(type))
        };
    }

    public Button CreateButton(ButtonType type, string name, int targetID)
    {
        return type switch
        {
            ButtonType.ActionButton => new ActionButton(name, navigatorReference, targetID),
            ButtonType.RedirectButton => new RedirectButton(name, navigatorReference, targetID),
            _ => throw new ArgumentOutOfRangeException(nameof(type))
        };
    }
}

abstract class Button
{
    protected int? targetMenuID;
    protected string ButtonName { get; protected set; }
    protected Action action;
    protected event Action<ConsoleKeyInfo> onKeyPressed;
    protected IMenuNavigator navigator;
    protected bool isSelected = false;


    protected Button(string name)
    {
        ButtonName = name;
        StartKeyListener();

        onKeyPressed += key =>
        {
            if (key.Key == ConsoleKey.Enter)
                if (isSelected) Execute();
        };
    }

    public void changeIsSelectedStateTo(bool state) => isSelected = state;

    protected void DoAction() => action?.Invoke();
    public static void StartKeyListener()
    {
        new Thread(() =>
        {
            while (true)
            {
                ConsoleKeyInfo key = Console.ReadKey(true);
                onKeyPressed?.Invoke(key);
            }
        }).Start();
    }


    public abstract void Execute();
}

class ActionButton : Button
{
    public ActionButton(string name, IMenuNavigator nav) : base(name)
    {
        navigator = nav;
    }
    public ActionButton(string name, IMenuNavigator nav, int targetID) : base(name)
    {
        navigator = nav;
        targetMenuID = targetID;
    }

    public void SetAction(Action action) => action += action;

    public override void Execute()
    {
        DoAction();
        if (targetMenuID.HasValue)
        {
            navigator.ShowMenu(targetMenuID.Value);
        }
    }
}

class RedirectButton : Button
{
    public RedirectButton(string name, IMenuNavigator nav, int targetID) : base(name)
    {
        targetMenuID = targetID;
        navigator = nav;
    }

    public override void Execute()
    {
        navigator.ShowMenu(targetMenuID.Value);
    }


}

class ExitButton : Button
{
    public ExitButton(string name, IMenuNavigator nav) : base(name)
    {
        navigator = nav;
    }

    public override void Execute()
    {
        Console.Clear();
        Environment.Exit(0);
    }
}

class SaveButton : Button
{
    public SaveButton(string name, IMenuNavigator nav) : base(name)
    {
        navigator = nav;
    }
    public override void Execute(Action action)
    {

    }
}



// class MenuButton : Button
// {
//     private event Action action;
//     private int? targetMenuID;

//     public MenuButton(string name) : base(name) { }

//     public void SetTargetMenu(int id) => targetMenuID = id;

//     public void SetAction(Action action) => action += action;
//     public override void Execute()
//     {
//         if (targetMenuID.HasValue)
//         {
//             OnAction?.Invoke();
//             navigator.ShowMenu(targetMenuID.Value);
//         }

//     }
// }



