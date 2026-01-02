using System;
using System.Collections.Generic;

public interface IMenuNavigator
{
    void ShowMenu(int id);
    void GoBack();
    void ExitProgram();
}

public interface IMenuRenderer
{
    void Clear();
    void DrawButtons(List<string> names, int selectedIndex);
}


public class MenuManager : IMenuNavigator
{
    private readonly Dictionary<int, MenuScreen> menus = new();
    private readonly Stack<int> history = new();
    private readonly IMenuRenderer renderer;

    private MenuScreen currentMenu;

    public MenuManager(IMenuRenderer renderer)
    {
        this.renderer = renderer;
    }

    public void AddMenu(MenuScreen menu) => menus[menu.GetID()] = menu;

    public void ShowMenu(int id)
    {
        if (currentMenu != null)
            history.Push(currentMenu.Id);

        if (!menus.TryGetValue(id, out currentMenu))
            throw new InvalidOperationException($"Menu ID {id} not found");

        currentMenu.Start(this, renderer);
    }

    public void GoBack()
    {
        if (history.Count == 0)
            return;

        int previousMenu = history.Pop();
        ShowMenu(previousMenu);
    }

    public void ExitProgram()
    {
        renderer.Clear();
        Environment.Exit(0);
    }
}

public class MenuScreen
{
    private int Id;
    private readonly List<Button> buttons = new();
    private int selectedIndex = 0;

    public MenuScreen(int id)
    {
        Id = id;
    }

    public void AddButton(Button button)
        => buttons.Add(button);

    public void Start(IMenuNavigator navigator, IMenuRenderer renderer)
    {
        bool running = true;

        while (running)
        {
            renderer.DrawButtons(GetButtonNames(), selectedIndex);

            ConsoleKeyInfo key = Console.ReadKey(true).Key;

            switch (key)
            {
                case ConsoleKey.UpArrow:
                    ChangeSelection(-1);
                    break;

                case ConsoleKey.DownArrow:
                    ChangeSelection(1);
                    break;

                case ConsoleKey.Enter:
                    buttons[selectedIndex].Execute(navigator);
                    running = false;
                    break;
            }
        }
    }

    private int GetID() => Id;

    private List<string> GetButtonNames()
    {
        List<string> names = new();
        foreach (Button button in buttons)
            names.Add(button.GetName());

        return names;
    }

    private void ChangeSelection(int dir)
    {
        selectedIndex = (selectedIndex + dir + buttons.Count) % buttons.Count;
    }
}


public class ConsoleRenderer : IMenuRenderer
{
    public void Clear() => Console.Clear();

    public void DrawButtons(List<string> names, int selectedIndex)
    {
        Clear();
        for (int i = 0; i < names.Count; i++)
        {
            if (i == selectedIndex)
                Console.WriteLine("> " + names[i]);
            else
                Console.WriteLine("  " + names[i]);
        }
    }
}

public abstract class Button
{
    private string Name;

    private event Action<IMenuNavigator>? OnPressed;

    protected Button(string name)
    {
        Name = name;
    }

    public void Execute(IMenuNavigator nav)
    {
        OnPressed?.Invoke(nav);
    }

    public void AddOnPressedSubscription(Action action) => OnPressed += action;
    public string GetName() => Name;
}

public class ActionButton : Button
{
    public ActionButton(string name) : base(name) { }
}

public class ExitButton : Button
{
    public ExitButton(string name) : base(name) { }
}

public class RedirectButton : Button
{
    private int targetId;

    public RedirectButton(string name, int targetId) : base(name)
    {
        targetId = targetId;
    }
    public int GetTargetId() => targetId;
}

public enum ButtonType
{
    Action,
    Exit,
    Redirect
}

public class ButtonFactory
{
    private readonly IMenuNavigator navigatorRef;

    public ButtonFactory(IMenuNavigator nav)
    {
        navigatorRef = nav;
    }

    public Button Create(ButtonType type, string name, Action? action = null, int? targetMenu = null)
    {
        Button button = type switch
        {
            ButtonType.Action   => new ActionButton(name),
            ButtonType.Exit     => new ExitButton(name),
            ButtonType.Redirect => new RedirectButton(name, targetMenu!.Value),
            _ => throw new NotImplementedException()
        };

        BindEvent(button, action, targetMenu);

        return button;
    }

    private void BindEvent(Button button, Action? action, int? targetMenu)
    {
        button.AddOnPressedSubscription(nav =>
        {
            if (action != null)
                action.Invoke();

            if (button is RedirectButton redirect)
                nav.ShowMenu(redirect.GetTargetId());

            if (button is ExitButton)
                nav.ExitProgram();
        });
    }
}
