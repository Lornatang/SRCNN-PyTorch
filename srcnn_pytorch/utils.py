# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
import time

TOTAL_BAR_LENGTH = 50
LAST_T = time.time()
BEGIN_T = LAST_T


def progress_bar(current_epoch, total_epoch, current_batch, total_batch, msg=None):
    """ A progress bar for training.

    Args:
        current_epoch (int): Number of current training epoch.
        total_epoch (int): Total number of training epochs required.
        current_batch (int): Number of current training batches.
        total_batch (int): Total number of training batches required.
        msg (str): Output information.

    Returns:
        sys.stdout.write().

    """
    global LAST_T, BEGIN_T
    if current_batch == 0:
        BEGIN_T = time.time()  # Reset for new bar.

    current_len = int(TOTAL_BAR_LENGTH * (current_batch + 1) / total_batch)
    rest_len = int(TOTAL_BAR_LENGTH - current_len) - 1

    sys.stdout.write(f"[{current_epoch + 1}/{total_epoch}][{total_batch + 1}/{total_batch}]")
    sys.stdout.write(" [")
    for i in range(current_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    current_time = time.time()
    step_time = current_time - LAST_T
    LAST_T = current_time
    total_time = current_time - BEGIN_T

    time_used = f"  Step time: {step_time:.2f}s"
    time_used += f" | Total time: {total_time:.2f}s"
    if msg:
        time_used += " | " + msg

    msg = time_used
    sys.stdout.write(msg)

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


# return the formatted time
def format_time(seconds):
    """ Format time

    Args:
        seconds (int): Run time in seconds.

    Returns:
        Output a unit with run time.

    """
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    seconds_final = int(seconds)
    seconds = seconds - seconds_final
    millis = int(seconds * 1000)

    output = ""
    time_index = 1
    if days > 0:
        output += str(days) + "D"
        time_index += 1
    if hours > 0 and time_index <= 2:
        output += str(hours) + "h"
        time_index += 1
    if minutes > 0 and time_index <= 2:
        output += str(minutes) + "m"
        time_index += 1
    if seconds_final > 0 and time_index <= 2:
        output += str(seconds_final) + "s"
        time_index += 1
    if millis > 0 and time_index <= 2:
        output += str(millis) + "ms"
        time_index += 1
    if output == "":
        output = "0ms"
    return output
