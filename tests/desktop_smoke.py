"""Prova visual/manual: creates the real window and exercises its main actions."""

import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desktop_app import MovieLabDesktop


def run_smoke_test() -> None:
    # Dialogs would stop an unattended smoke test.
    messagebox.showinfo = lambda *args, **kwargs: None
    messagebox.showwarning = lambda *args, **kwargs: None
    messagebox.showerror = lambda *args, **kwargs: None

    root = tk.Tk()
    root.withdraw()
    app = MovieLabDesktop(root)
    print("Ventana creada", flush=True)
    assert app.root.tk.call("ttk::style", "lookup", "Treeview", "-foreground") == "#111827"
    assert app.new_genres_list.cget("foreground") == "#111827"
    app._show_new_user_mode("Avançat")
    assert app.new_user_mode.get() == "Avançat"
    app._show_new_user_mode("Senzill")
    initial_users = app.ratings["userId"].nunique()
    app.optimization_trials.set(10)
    phase = {"value": 0}

    def check_progress():
        if app.busy:
            root.after(50, check_progress)
            return
        if phase["value"] == 0:
            print("Comparació acabada", flush=True)
            assert app.results is not None and len(app.results) == 5
            phase["value"] = 1
            app.generate_recommendations()
            root.after(50, check_progress)
        elif phase["value"] == 1:
            print("Recomanacions terminadas", flush=True)
            assert app.recommendations is not None and len(app.recommendations) == 10
            phase["value"] = 2
            app.optimize_hybrid()
            root.after(50, check_progress)
        elif phase["value"] == 2:
            print("Optimització acabada", flush=True)
            weights = [variable.get() for variable in app.hybrid_weights.values()]
            assert abs(sum(weights) - 1.0) < 0.01
            titles = list(app.simple_movie_combos[0]["values"])
            for index, title in enumerate(titles[:4]):
                app.simple_movie_vars[index].set(title)
            app.simple_rating_vars[0].set("Em va encantar")
            app.simple_rating_vars[1].set("Em va agradar")
            app.simple_rating_vars[2].set("No em va agradar")
            app.simple_rating_vars[3].set("Em va encantar")
            if "Adventure" in app.simple_genre_vars:
                app.simple_genre_vars["Adventure"].set(True)
            phase["value"] = 3
            app._generate_simple_recommendations()
            root.after(50, check_progress)
        else:
            print("Nou usuari recomendado", flush=True)
            assert app.new_user_recommendations is not None
            assert len(app.new_user_recommendations) == 10
            app.save_new_user()
            assert app.ratings["userId"].nunique() == initial_users + 1
            phase["value"] = 4
            root.destroy()

    app.run_comparison()
    root.after(50, check_progress)
    root.mainloop()
    assert phase["value"] == 4
    print("Prova de escritorio completada: comparació, recomanació, optimització i alta de usuari.")


if __name__ == "__main__":
    run_smoke_test()
