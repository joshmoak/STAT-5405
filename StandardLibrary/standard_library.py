# standard_library.py
"""Python Essentials: The Standard Library.
<Name>
<Class>
<Date>
"""

import calculator as cal
import box
from itertools import chain, combinations
import time
from random import randint

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    print(min(L), max(L), (sum(L)/len(L)), sep = ",")

    #raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    int_1 = 5
    int_2 = int_1
    int_2 = 3
    if int_1 == int_2:
        print("Numbers are mutable")
    else:
        print("Numbers are immutable")

    str_1 = "test"
    str_2 = str_1
    str_2 = "testing"
    if str_1 == str_2:
        print("Strings are mutable")
    else:
        print("Strings are immutable")

    list_1 = [1,2,3,4,5]
    list_2 = list_1
    list_2[0] = 0
    if list_1 == list_2:
        print("Lists are mutable")
    else:
        print("Lists are immutable")

    tup_1 = (1,2,3,4,5)
    tup_2 = tup_1
    tup_2 = (0,2,3,4,5)
    if tup_1 == tup_2:
        print("Tuples are mutable")
    else:
        print("Tuples are immutable")

    set_1 = {1,2,3,4,5}
    set_2 = set_1
    set_2 = {0,2,3,4,5}
    if set_1 == set_2:
        print("Sets are mutable")
    else:
        print("Sets are immutable")
    

    #raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt that are 
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    hypotenuse = cal.sqrt(cal.sum(cal.product(a,a),cal.product(b,b)))
    return hypotenuse
    # raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """

    s = list(A)
    results = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    return results


    #raise NotImplementedError("Problem 4 Incomplete")


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    player = input("Enter your name: \n")
    print(f'Player name is {player}')
    timelimit = input("Enter time limit in seconds: \n")
    print(f'Time limit is {timelimit} seconds')
    still_playing = true
    remaining = [1,2,3,4,5,6,7,8,9]

    while still_playing = true:
        time_1 = time.time()
        a, b = randint(1,6), randint(1,6)
        if sum(remaining <= 6):
            b = 0
        roll = a + b
        print("Numbers left: " + remaining "\n")
        print("Roll: " + roll "\n")
        print("Seconds left: " + timelimit "\n")
        still_playing = box.isvalid(roll, remaining)
        time_2 = time.time()
        time_left = timelimit - (time_2 - time_1)

        if time_left <= 0 OR box.isvalid(roll,remaining) = false:
            still_playing = false
            print("You loose, say something about the score")

        


        


    box.parse_input(player_input, remaining)

    print("Numbers to eliminate: " + remaining "\n\n")