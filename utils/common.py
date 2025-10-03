def hms_string(sec):
    """
    Convert seconds to a formatted string (hours:minutes:seconds)
    
    Args:
        sec: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = int(sec % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
