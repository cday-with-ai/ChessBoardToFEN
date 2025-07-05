from typing import List, Tuple


def board_array_to_fen(board: List[List[str]], 
                      turn: str = 'w',
                      castling: str = 'KQkq',
                      en_passant: str = '-',
                      halfmove: int = 0,
                      fullmove: int = 1) -> str:
    """
    Convert 8x8 board array to FEN string
    
    Args:
        board: 8x8 array of piece symbols ('K', 'Q', 'R', 'B', 'N', 'P' for white,
               'k', 'q', 'r', 'b', 'n', 'p' for black, 'empty' for empty squares)
        turn: 'w' for white, 'b' for black
        castling: Castling rights (e.g., 'KQkq', '-')
        en_passant: En passant target square (e.g., 'e3', '-')
        halfmove: Halfmove clock
        fullmove: Fullmove number
    
    Returns:
        FEN string
    """
    fen_rows = []
    
    for row in board:
        fen_row = ""
        empty_count = 0
        
        for square in row:
            if square == 'empty' or square == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += square
        
        # Add remaining empty squares count
        if empty_count > 0:
            fen_row += str(empty_count)
        
        fen_rows.append(fen_row)
    
    # Join rows with '/'
    position = '/'.join(fen_rows)
    
    # Construct full FEN
    fen_parts = [position, turn, castling, en_passant, str(halfmove), str(fullmove)]
    return ' '.join(fen_parts)


def squares_to_board_array(squares: List[Tuple[str, float]], 
                          confidence_threshold: float = 0.5) -> List[List[str]]:
    """
    Convert list of classified squares to 8x8 board array
    
    Args:
        squares: List of (piece, confidence) tuples in FEN order (a8 to h1)
        confidence_threshold: Minimum confidence to accept classification
    
    Returns:
        8x8 board array
    """
    board = []
    
    for rank in range(8):
        row = []
        for file in range(8):
            idx = rank * 8 + file
            piece, confidence = squares[idx]
            
            # Use empty if confidence is too low
            if confidence < confidence_threshold:
                row.append('empty')
            else:
                row.append(piece)
        
        board.append(row)
    
    return board


def build_fen_from_squares(squares: List[Tuple[str, float]], 
                          confidence_threshold: float = 0.5,
                          include_full_fen: bool = True) -> str:
    """
    Build FEN string from classified squares
    
    Args:
        squares: List of (piece, confidence) tuples
        confidence_threshold: Minimum confidence threshold
        include_full_fen: If True, include turn/castling/etc. If False, just position
    
    Returns:
        FEN string
    """
    board_array = squares_to_board_array(squares, confidence_threshold)
    
    if include_full_fen:
        return board_array_to_fen(board_array)
    else:
        # Just return the position part
        fen_rows = []
        for row in board_array:
            fen_row = ""
            empty_count = 0
            
            for square in row:
                if square == 'empty':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += square
            
            if empty_count > 0:
                fen_row += str(empty_count)
            
            fen_rows.append(fen_row)
        
        return '/'.join(fen_rows)


def validate_fen_position(fen_position: str) -> Tuple[bool, str]:
    """
    Validate just the position part of a FEN string
    
    Returns:
        (is_valid, error_message)
    """
    rows = fen_position.split('/')
    
    if len(rows) != 8:
        return False, f"Expected 8 rows, got {len(rows)}"
    
    for i, row in enumerate(rows):
        file_count = 0
        for char in row:
            if char.isdigit():
                file_count += int(char)
            elif char in 'KQRBNPkqrbnp':
                file_count += 1
            else:
                return False, f"Invalid character '{char}' in row {i+1}"
        
        if file_count != 8:
            return False, f"Row {i+1} has {file_count} squares, expected 8"
    
    # Check king counts
    white_kings = fen_position.count('K')
    black_kings = fen_position.count('k')
    
    if white_kings != 1:
        return False, f"Expected 1 white king, found {white_kings}"
    if black_kings != 1:
        return False, f"Expected 1 black king, found {black_kings}"
    
    return True, "Valid"