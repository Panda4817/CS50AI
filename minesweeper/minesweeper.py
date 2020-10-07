import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        # If the count is the same as the number of cells, then all cells are mines
        if len(self.cells) == self.count:
            return self.cells.copy()
        
        return None

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        # If the count is 0, it means all the cells are safe
        if self.count == 0:
            return self.cells.copy()
        return None

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        # Check cell in sentence and sentence has more than  count 0
        if cell in self.cells and self.count > 0:
            # Remove cell from sentence as we know it is a mine and reduce the count
            self.cells.remove(cell)
            self.count -= 1
        return

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        # Check cell is in sentence and count is more than 0
        if cell in self.cells and self.count > 0:
            # remove cell from sentence as we know it is safe
            # nothing happens to count
            self.cells.remove(cell)
        return


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)
    
    def knowledge_check(self):
        """
        Loop over knowledge base and check for known mines and known safes
        """
        # Loop though sentences and find safe cells
        for sentence in self.knowledge:
            safes = sentence.known_safes()
            if safes != None:
                for cell in safes:
                    self.mark_safe(cell)
        # Loop through sentences and find cells with mines   
        for sentence in self.knowledge:
            mines = sentence.known_mines()
            if mines != None:
                for cell in mines:
                    self.mark_mine(cell)
            
        return
    
    def find_sentences(self):
        """
        Loop over knowledge base twice, checking for subsets and if new sentences can be added
        """
        # Initialise an empty list to store new sentences
        new_knowledge = []

        # Loop over knowledge base twice
        for sentence1 in self.knowledge:
            for sentence2 in self.knowledge:
                # Check if one sentence is a subset of the other
                if sentence1.cells < sentence2.cells:
                    # If so, find the difference in cells
                    new_cells = sentence2.cells.difference(sentence1.cells)
                    # Find the difference in count
                    new_count = sentence2.count - sentence1.count
                    # Store new sentence object in a variable
                    s = Sentence(new_cells, new_count)
                    # Check that new sentence is not already in the knowledge base (prevent infinite loop)
                    if s not in self.knowledge:
                        # Append new sentence to list
                        new_knowledge.append(s)
        # If the list length is more than 0, return the list
        if len(new_knowledge) > 0:
            return new_knowledge
        
        # Else return None
        return None

    def get_cells(self, cell):
        """
        From a given safe cell, find the coordinates of all cells around it
        """
        # An empty list to store the tuples of cells
        cells = []

        # Loop though from row above to row below
        for row in range(cell[0] - 1, cell[0] + 2):
            # Check the row  is within board
            if row < 0 or row >= self.height:
                continue
            # Loop through from column before to column after
            for col in range(cell[1] - 1, cell[1] + 2):
                # Check the column is within the board
                if col < 0 or col >= self.width:
                    continue
                # Check we are not counting the original cell
                if (row, col) == cell:
                    continue
                # Append new coordinate to list
                cells.append((row, col))
        # Return the list to be used in a sentence
        return cells
    
    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # Mark the cell as a move made
        self.moves_made.add(cell)
        
        # Mark the cell as safe
        self.mark_safe(cell)
        
        # Get all the cells around this cell
        cells = self.get_cells(cell)
        
        # Create a new sentence object and append it into knowledge base
        self.knowledge.append(Sentence(cells, count))
        
        # Check through the knowledge base and mark cells safe or mines if it can be done
        self.knowledge_check()
        
        # Initialise a list to store new sentences
        new_knowledge = []

        # Find new sentences by checking for subsets
        x = self.find_sentences()
        # Check None is note returned before appending it to the list
        if x != None:
            new_knowledge.extend(x)

        # While loop used to keep making inferences until it new sentences are not found
        while len(new_knowledge) != 0:
            # Add new sentences to knowledge base
            self.knowledge.extend(new_knowledge)
            # Check through the KB for safe cells and mines
            self.knowledge_check()
            # Clear the list for new sentences to be added
            new_knowledge.clear()
            # Find new sentences again by checking for subsets
            y = self.find_sentences()
            # Check None is not returned before appending it to the new_knowledge list
            if y != None:
                new_knowledge.extend(y)

        return

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        # find the difference between safe cells and moves made to find cells that safe but not played yet
        safe_moves = self.safes - self.moves_made
        # If this new set is not empty, return one of the cells
        if len(safe_moves) > 0:
            move = safe_moves.pop()
            return move
        # Else return Noneto indicate no safe moves
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        # Initialise an empty set to store random cells
        rand_cells = set()
        # Loop through height numbers and row numbers
        for i in range(self.height):
            for j in range(self.width):
                # Store the coordinate as a tuple
                maybe = (i, j)
                # Check this tuple is not a mine and not a move already made
                if maybe not in self.mines and maybe not in self.moves_made:
                    # Add it to set as a move that could be made
                    rand_cells.add(maybe)
        # If the set is not empty, return a random cell from that set
        if len(rand_cells) > 0:
            return rand_cells.pop()
        # Else return None indicating no moves left
        return None
