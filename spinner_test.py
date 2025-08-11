import time
import threading
import functools
from tqdm import tqdm


# clearColor = "\033[0m"
# spinner_chars = " घूम " # Some example unicode characters
# with tqdm(total=None, bar_format="{desc}") as pbar:
#     for i in range(50):  # Simulate a task with 50 steps
#         # Your long-running task goes here
#         time.sleep(0.1)
        
#         # Update the spinner character in the description
#         pbar.set_description_str(f"{deepPurple}Aligning Trajectory... {pink}{spinner_chars[i % len(spinner_chars)]}{clearColor}")



def bouncing_bar_decorator(bar_width=40, ball_text="Processing..."):
    """
    A decorator that displays a bouncing bar animation while the decorated
    function is executing.

    Args:
        bar_width (int): The visual width of the bouncing bar.
        ball_text (str): The text that "bounces" from side to side.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # --- Setup for capturing the function's return value ---
            result_container = {"result": None}

            def task_wrapper():
                """A helper to run the function and store its result."""
                result_container["result"] = func(*args, **kwargs)

            # --- Threading setup ---
            task_thread = threading.Thread(target=task_wrapper)
            task_thread.start()

            # --- Animation setup ---
            deepPurple = "\033[35m"
            resetColor = "\033[0m"
            ball = f"{deepPurple}{ball_text}{resetColor}"
            ball_len = len(ball_text)

            # --- TQDM Animation Loop ---
            t = 0
            with tqdm(total=None, bar_format="{desc}", leave=False) as pbar:
                while task_thread.is_alive():
                    effective_width = bar_width - ball_len
                    if effective_width <= 0: # Handle cases where text is too long
                        pbar.set_description_str(ball)
                        continue

                    # Bouncing position calculation
                    position = t % (2 * effective_width)
                    if position > effective_width:
                        position = 2 * effective_width - position

                    # Create the bar string
                    bar_str = ' ' * position + ball + ' ' * (effective_width - position)
                    pbar.set_description_str(f"[{bar_str}]")
                    
                    time.sleep(0.05)
                    t += 1

            # Clean up: wait for the thread and return the captured result
            task_thread.join()
            return result_container["result"]
        return wrapper
    return decorator



@bouncing_bar_decorator(bar_width=60, ball_text="Aligning Trajectory")
def long_running_task():
    """Simulates a task that takes a few seconds."""
    time.sleep(6)

# --- Run it ---
if __name__ == "__main__":
    long_running_task()
