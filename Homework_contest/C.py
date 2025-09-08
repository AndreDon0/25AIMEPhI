class Leg:
    def __init__(self):
        pass

class Animal:
    from typing import List
    def __init__(self, legs: List[Leg]):
        self.legs = legs
        self.n_legs = len(legs)

class Cat(Animal):
    def __init__(self):
        # Almost all cats have 4 legs exept rare Schrodinger's cat that has undefined number of legs.
        super().__init__([Leg() for _ in range(4)])

    def speak(self):
        return "Meow!"

class Goose(Animal):
    def __init__(self):
        # As Encyclopedia denote "A goose (pl.: geese) is a bird of any of several waterfowl species in the family Anatidae and has got two legs."
        super().__init__([Leg() for _ in range(2)])

    def speak(self):
        return "Honk!"

class GooseCatGroup:
    def __init__(self, n_legs: int):
        if n_legs < 0 or n_legs % 2 == 1:
            print(0)
            exit(0)
            raise Exception("There's Schrodinger's cat nearby hence this group cannot be defined properly")

        # Since we don't know number of heads this group feels unstable and can dissepate every second.
        # Let's stabilize it by filling it with gooses!

        self.animals = [Goose() for _ in range(n_legs // 2)]


    # After a nice shake couple of gooses drop off and one cat pops out.
    # Warning: You can shake a groop only n_legs // 4 occasions! Owerwise Schrodinger's cat collapse a group!
    def shake(self):
        for i in range(len(self.animals) - 1):
            if isinstance(self.animals[i], Goose) and isinstance(self.animals[i + 1], Goose):
                self.animals[i:i + 2] = [Cat()]
                return None
        
        raise Exception("You shake too many times therefore Schrodinger's cat nearby hence this group cannot be defined properly")
    
    def __str__(self):
        n_gooses = sum([type(animal) == Goose for animal in self.animals])
        n_cats = sum([type(animal) == Cat for animal in self.animals])
        return f"{n_gooses} {n_cats}"
        return f"Gooses: {n_gooses}, cats: {n_cats}"


group = GooseCatGroup(int(input()))
while True:
    try:
        print(group)
        group.shake()
    except Exception:
        exit(0)