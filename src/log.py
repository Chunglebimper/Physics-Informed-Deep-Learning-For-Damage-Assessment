class Log:
    def __init__(self, path):
        self.path = path
        self.log_file = None
        self.run_number = 1

    def open(self):
        """
        * creates new or opens existing file
        * automatic run detection
        """
        self.log_file = open(self.path, "a")
        # Try to read the last line to find last run number
        with open(self.path, 'r') as f:
            #
            lines = f.readlines()
            for line in reversed(lines):    #for each reveresed line
                if line.strip().startswith("-> Run #"):
                    try:
                        self.run_number = int(line.strip().split('#')[-1]) + 1
                        break
                    except ValueError:
                        pass  # malformed line, skip
            self.append(f"-> Run #{self.run_number}")

    def close(self):
        """
        closes file
        """
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def append(self, to_be_logged):
        """
        appends data to file
        """
        if self.log_file:
            self.log_file.write(to_be_logged + '\n')
            self.log_file.flush()
        else:
            raise RuntimeError("Log file is not open. Call `open()` first.")