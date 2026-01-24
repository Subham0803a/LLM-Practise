# def hello(name, a, b):
#     print(f"Hello, {name}!")
#     return a + b

# result = hello("subham", 5, 10)
# print("The sum is:", result)

# -------------------------------------------------

# def addition(x, y):
#     print("This is addition function")
#     return x + y

# def subtraction(x, y):
#     print("This is subtraction function")
#     return x - y

# def multiplication(x, y):
#     print("This is multiplication function")
#     return x * y

# def division(x, y):
#     print("This is division function")
#     if y == 0:
#         return "Cannot divide by zero"
#     return x / y

# def calculator():
#     name = input("Enter your name:- ")
    
#     print("\nChoose operation:")
#     print("1. Addition")
#     print("2. Subtraction")
#     print("3. Multiplication")
#     print("4. Division")
    
#     choice = input("Enter choice (1/2/3/4):- ")
    
#     x = float(input("Enter first number: "))
#     y = float(input("Enter second number: "))
    
#     if choice == "1":
#         result = addition(x, y)
#     elif choice == "2":
#         result = subtraction(x, y)
#     elif choice == "3":
#         result = multiplication(x, y)
#     elif choice == "4":
#         result = division(x, y)
#     else:
#         print("Invalid choice")
#         return

#     print(f"{name} The result is: {result}")
    
# calculator()

# ----------------------------------------------------

# def odd_even(num):
#     if num % 2 == 0:
#         return "Even"
#     else:           
#         return "Odd"
    
# number = int(input("Enter a number: "))
# data = odd_even(number)
# # print(f"The number {number} is {odd_even(number)}.")
# print(f"The number {number} is {data}.")

# -------------------------------------------------------

# def addition(a, b):
#     return a + b

# def subtract(a, b):
#     return a - b

# def multiply(a, b):
#     return a * b

# def divide(a, b):
#     if b == 0:
#         return "Cannot divide by zero"
#     return a / b

# def calculator():
#     print("\nChoose operation:")
#     print("1. Addition")
#     print("2. Subtraction")
#     print("3. Multiplication")
#     print("4. Division")
    
#     choice = input("Enter choice (1/2/3/4):- ")
    
#     x = float(input("Enter first number: "))
#     y = float(input("Enter second number: "))
    
#     # if choice == "1":
#     #     result = addition(x, y)
#     # elif choice == "2":
#     #     result = subtract(x, y)
#     # elif choice == "3":
#     #     result = multiply(x, y)
#     # elif choice == "4":
#     #     result = divide(x, y)
#     # else:
#     #     print("Invalid choice")
#     #     return
    
#     # print(f"The result is: {result}")
    
#     if choice == "1":
#         print(f"The result is: {addition(x, y)}")
#     elif choice == "2":
#         print(f"The result is: {subtract(x, y)}")
#     elif choice == "3":
#         print(f"The result is: {multiply(x, y)}")
#     elif choice == "4":
#         print(f"The result is: {divide(x, y)}")
#     else:
#         print("Invalid choice")
#         return
    

# calculator()
