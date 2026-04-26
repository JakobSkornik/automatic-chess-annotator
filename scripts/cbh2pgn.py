#!/usr/bin/env python3

# https://codeberg.org/FelixKling/cbh2pgn/src/branch/main/cbh2pgn.py

# -*- coding: utf-8 -*-

# A script for reading Chessbase cbh databases in Python.
# Copyright (C) 2023 Dr. Felix Kling, Mainz

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/

import os 
import chess
import chess.pgn
import io
import numpy as np

def read_cbh(file_cbh, max_games=1e20, output="pgn", encoding="auto"):
    """
    This function reads a Chessbase .cbh database.

    Parameters
    ----------
    file_cbh : str
        The file path to the .cbh file.
    max_games : number, optional
        Determine the number of games up to which the database is converted. The default is 1e20.
    output : str
        Type of the return value.
        "pgn" is a pgn string
        "pychess" is a list of chess.pgn.Game objects
        "write" writes a pgn file with the same name as the original database. Any existing file is overwritten.
    encoding : str
        The encoding used for text.
        "auto" is a primitive automatic detection.
        "cp1252" was the default before.
    Returns
    -------
    Depending on output argument: A pgn string or a list of chess.pgn.Game() objects or None.


    """
    
    # %%
    assert file_cbh.lower().endswith(".cbh"), "Invalid file extension! Please provide a .cbh file!"
    
    f_path = file_cbh.rsplit("/", 1)[0] + "/"
    files = {f.rsplit(".", 1)[1].lower(): f_path + f for f in os.listdir(f_path) if f.startswith(file_cbh[:-4].rsplit("/", 1)[1])}
 
    #% following http://talkchess.com/forum3/viewtopic.php?topic_view=threads&p=287896&t=29468

    ann_types = {
        b"\x02": "text after move",
        b"\x03": "symbols",
        b"\x07": "?",
        b"\x18": "critical position",
        b"\x82": "text before move",
        b"\x14": "pawn structure", #(ofs 6 = 3?! TODO)
        b"\x15": "piece path", #(ofs 6-7 = 03 2b? 03 0d? TODO)
        b"\x13": "game quotation",
        b"\x22": "medal",
        b"\x23": "variation color",
        b"\x10": "sound",
        b"\x20": "video",
        b"\x11": "picture",
        b"\x09": "training annotation", # (TODO)
        b"\x61": "correspondence header", # (always move -1?)
        b"\x19": "correspondence move", # (header must exist?)
        b"\x24": "?", #always len 39 - Time control?
        b"\x69": "?",
    }
    
    move_comments = {
     b'\x00': '',
     b'\x01': '$1',
     b'\x02': '$2',
     b'\x03': '$3',
     b'\x04': '$4',
     b'\x05': '$5',
     b'\x06': '$6',
     b'\x07': '$7',
     b'\x08': '$7',
     b'\x0a': '$10',
     b'\x0b': '$10',
     b'\x0d': '$13',
     b'\x0e': '$14',
     b'\x0f': '$15',
     b'\x10': '$16',
     b'\x11': '$17',
     b'\x12': '$18',
     b'\x13': '$19',
     b'\x16': '$22',
     b'\x17': '$23',
     b'\x18': '$24',
     b'\x20': '$32',
     b'\x24': '$36',
     b'\x28': '$40',
     b'\x2c': '$44',
     b'\x2d': '$45',
     b'\x82': '$130',
     b'\x83': '$131',
     b'\x84': '$132',
     b'\x85': '$133',
     b'\x88': "$138",
     b'\x8a': '$139',
     b'\x8c': '$140',
     b'\x8d': '$141',
     b'\x8e': '$142',
     b'\x8f': '$143',
     b'\x90': '$144',
     b'\x91': '$145',
     b'\x92': '$146',
     }

    global special_chars
    special_chars = {
    b'\x8c': '∆',
    b'\x8d': '∇',
    b'\x8e': '⌓',
    b'\x8f': '≤',
    b'\x90': '=',
    b'\x91': 'RR',
    b'\x92': 'N',
    b"\xa7": "♙",
    b"\xa6":"♖",
    b"\xa5":"♗",
    b"\xa4": "♘",
    b"\xa3":"♕",
    b"\xa2": "♔",
    b"\x9e": "🙾",        
    b'\x82': "→",
    b'\x83': "↑",
    b'\x84': "⇆",
    b'\x85': "∆",
    b'\x86': "○",
    b'\x87': "⨀",
    b'\x90': "⟳",
    b'\x91': "⇔",
    b'\x92': "⇗",
    b'\x93': "⨁",
    b'\x94': "⊞",
    b'\x99': "□",
    b'\xa9': "=/∞",
    b'\xaa': "└",
    b'\xab': "⟪",
    b'\xac': "⊥",
    b'\xad': "⯴",
    b'\xae': "⯵",
    b'\xaf': "⯶",
    b'\xb0': "=/∞",
    b'\xb1': "±",
    b'\xb2': "⩲",
    b'\xb3': "⩱",
    b'\xb5': "∓",
    b'\xb9': "⌓",
    b'\xba': "┘",
    b'\xbb': "⟫",
    b'\xd7': "✕",
    b'\xf7': "∞",
    b'\xfe': "♂️",   
    }

    special_chars_cyrillic = {
    b'\x81': '∇',
    b'\x82': "→",
    b'\x83': "↑",
    b'\x84': "⇆",
    b'\x85': "∆",
    b'\x86': "○",
    b'\x87': "⨀",
    b'\x89': "⟳",
    b'\x8b': '≤',
    b'\x91': '⇔',
    b'\x92': "⇗",
    b'\x93': "⨁",
    b'\x94': "⊞",
    b'\x99': "□",
    b'\x9b': '≥',
    b'\x9e': '🙾',
    b'\xa1': '♔',
    b'\xa2': '♕',
    b'\xa3': '♘',
    b'\xa4': '♗',
    b'\xa5': '♖',
    b'\xa6': '♙',
    b'\xa8': '=/∞',
    b'\xa9': '└',
    b'\xaa': '⟪',
    b'\xab': '⟪',
    b'\xac': '⊥',
    b'\xad': '⯴',
    b'\xae': '⯵',
    b'\xaf': '⯶',
    b'\xb0': "=/∞",
    b'\xb1': "±",
    b'\xb2': "⩲",
    b'\xb3': "⩱",
    b'\xb5': "∓",
    b'\xb9': "⌓",
    b'\xba': "┘",
    b'\xbb': "⟫",
    }
    
    result_map = {
    3: "line",
    2: "1-0",
    1: "1/2-1/2",
    0: "0-1",
    6: "+:-",
    5: "=:=",
    4: "-:+",
    7: "0-0",
    }  
    
    global detected_encoding
    detected_encoding = encoding
    if encoding == "auto":
        detected_encoding = "cp1252"

    # cyrillic letters in cp1251
    CYRILLIC = {a for a in 
                {*range(188, 256),
                175, *range(178, 180), 184, 186,
                168, 170,
                *range(161, 164), 165,
                154, *range(156, 160),
                *range(140, 145),
                128, 129, 131, 138,
                } 
                if a.to_bytes(1, "big") not in special_chars_cyrillic}
    

    def decode_text(byte_string):
        """
        Function to decode Chessbase texts.

        Parameters
        ----------
        byte_string : bytes
            The input text as bytes.

        Returns
        -------
        result : str
            The decoded text.

        """
        global detected_encoding
        global special_chars
        # try to detect the correct encoding
        if encoding == "auto":
            if len(byte_string) > 0:
                # test for cyrillic letters, then cp1251
                fraction = len({a for a in byte_string if a in CYRILLIC}) / len(byte_string)
                # more than 20%: likely cyrillic
                if fraction > 0.2:
                    # cyrillic encoding
                    detected_encoding = "cp1251"
                    # new special symbols
                    special_chars = special_chars_cyrillic

        result = ""
        for byte in byte_string:
            byte = bytes([byte])
            if byte in special_chars:
                result += special_chars[byte]
            else:
                try:
                    result += byte.decode(detected_encoding)
                except UnicodeDecodeError:
                    pass
        return result
    
    # % read players        
    # cbp_path = files["cbp"]
    def read_cbp(cbp_path):
        """
        Function that reads a .cbp player file.

        Parameters
        ----------
        cbp_path : str
            The file path.

        Returns
        -------
        cbp : list
            A list of players.

        """
        with open(cbp_path, "rb") as cbp_file:
            cbp = []
            cbp_text = cbp_file.read()
            # looks like there is this variable offset telling us where the player names begin
            offset = int.from_bytes(cbp_text[24:28], byteorder="little", signed=False)
            # now the player names are stored in a tree, but we just need the names...
            # the header ends at byte 28 + offset, a player entry has a length of 67
            i = 0
            while i*67 + 28 + offset + 67 <= len(cbp_text):
                line = cbp_text[28 + i*67 + offset: 28 + i*67 + 67 + offset]
        
                last_name =  decode_text(line[9:39].split(b'\x00')[0])
                first_name =  decode_text(line[39:59].split(b'\x00')[0])                       

                cbp.append(f"{last_name}, {first_name}".strip(", "))
                i += 1
        return cbp
    
    # read annotators
    # cbc_path = files["cbc"]
    def read_cbc(cbc_path):
        """
        Function that reads a .cbc source file.

        Parameters
        ----------
        cbc_path : str
            The file path.

        Returns
        -------
        cbc : list
            A list of sources.

        """
        with open(cbc_path, "rb") as cbc_file:
            cbc = []
            cbc_text = cbc_file.read()
            # looks like there is this variable offset telling us where the names begin
            offset = int.from_bytes(cbc_text[24:28], byteorder="little", signed=False)
            len_entry = 62
            # now the names are stored in a tree, but we just need the names...
            # the header ends at byte 28 + offset, a player entry has a length of 67
            i = 0
            while i * len_entry + 28 + offset + len_entry <= len(cbc_text):
                line = cbc_text[28 + i * len_entry + offset: 28 + i * len_entry + len_entry + offset]
        
                name =  decode_text(line[9:54].split(b'\x00')[0])                    

                cbc.append(name.strip(", "))
                i += 1
        return cbc
    
    # read sources
    # cbs_path = files["cbs"]
    def read_cbs(cbs_path):
        """
        Function that reads a .cbs source file.

        Parameters
        ----------
        cbs_path : str
            The file path.

        Returns
        -------
        cbs : list
            A list of sources.

        """
        with open(cbs_path, "rb") as cbs_file:
            cbs = []
            cbs_text = cbs_file.read()
            # looks like there is this variable offset telling us where the names begin
            offset = int.from_bytes(cbs_text[24:28], byteorder="little", signed=False)
            len_entry = 68
            # now the names are stored in a tree, but we just need the names...
            # the header ends at byte 28 + offset, a player entry has a length of 67
            i = 0
            while i * len_entry + 28 + offset + len_entry <= len(cbs_text):
                line = cbs_text[28 + i * len_entry + offset: 28 + i * len_entry + len_entry + offset]
        
                name =  decode_text(line[9:34].split(b'\x00')[0])                    

                cbs.append(name.strip(", "))
                i += 1
        return cbs
    
    # read teams
    # cbe_path = files["cbe"]
    def read_cbe(cbe_path):
        """
        Function that reads a .cbe source file.

        Parameters
        ----------
        cbe_path : str
            The file path.

        Returns
        -------
        cbe : list
            A list of sources.

        """
        with open(cbe_path, "rb") as cbe_file:
            cbe = []
            cbe_text = cbe_file.read()
            # looks like there is this variable offset telling us where the names begin
            offset = int.from_bytes(cbe_text[24:28], byteorder="little", signed=False)
            len_entry = 72
            # now the names are stored in a tree, but we just need the names...
            # the header ends at byte 28 + offset, a player entry has a length of 67
            i = 0
            while i * len_entry + 28 + offset + len_entry <= len(cbe_text):
                line = cbe_text[28 + i * len_entry + offset: 28 + i * len_entry + len_entry + offset]
        
                name = decode_text(line[9:54].split(b'\x00')[0])                    
                # first = int.from_bytes(line[68:72], signed=False, byteorder="little")
                cbe.append(name.strip(", "))
                i += 1
        return cbe
    
    
        # % read tournaments        
        
    # cbt_path = files["cbt"]
    def read_cbt(cbt_path):
        """
        Function that reads a .cbt source file.

        Parameters
        ----------
        cbt_path : str
            The file path.

        Returns
        -------
        cbt : list
            A list of tournaments.

        """
        with open(cbt_path, "rb") as cbt_file:
            cbt = []
            cbt_text = cbt_file.read()
            # looks like there is this variable offset telling us where the player names begin
            offset = int.from_bytes(cbt_text[24:28], byteorder="little", signed=False)
            len_entry = 99
            # the header ends at byte 28 + offset, a tournament entry has a length of 67
            i = 0
            while i * len_entry + 28 + offset + len_entry <= len(cbt_text):
                line = cbt_text[28 + i * len_entry + offset: 28 + i * len_entry + len_entry + offset]
        
                event = decode_text(line[9:49].split(b'\x00')[0]) 
                site = decode_text(line[49:79].split(b'\x00')[0])
                            
                cbt.append({
                    "Event": event,
                    "Site": site,
                    })
                i += 1
        return cbt
    
            
    # cbj_path = files["cbj"]
    def read_cbj(cbj_path):
        """
        Function that reads a .cbj header file.

        Parameters
        ----------
        cbj_path : str
            The file path.

        Returns
        -------
        cbj : list
            A list of team indices (white, black).

        """
        with open(cbj_path, "rb") as cbj_file:
            cbj = []
            cbj_text = cbj_file.read()
            
            # looks like there is this variable offset telling us where the names begin
            offset = int.from_bytes(cbj_text[24:28], byteorder="little", signed=False)
            len_entry = 78
            # now the names are stored in a tree, but we just need the names...
            # the header ends at byte 28 + offset, a player entry has a length of 67
            i = 0
            while i * len_entry + 28 + offset + len_entry <= len(cbj_text):
                line = cbj_text[28 + i * len_entry + offset: 28 + i * len_entry + len_entry + offset]
                team_white = int.from_bytes(
                    line[4:8], byteorder="big", signed=True)
                team_black = int.from_bytes(
                    line[8:12], byteorder="big", signed=True)
                # data = []
                # data.append(int.from_bytes(line[0:4], byteorder="big", signed=True)) # 0 for some tournaments not null? first game?
                # data.append(int.from_bytes(line[4:8, byteorder="big", signed=True)) # 1 team white
                # data.append(int.from_bytes(line[8:12], byteorder="big", signed=True)) # 2 team black
                # data.append(int.from_bytes(line[12:16], byteorder="big", signed=True)) # 3 always null?
                # data.append(int.from_bytes(line[16:20], byteorder="big", signed=True)) # 4 always 0?
                # data.append(int.from_bytes(line[20:24], byteorder="big", signed=True)) # 5 cba start position in bytes
                # data.append(int.from_bytes(line[24:26], byteorder="big", signed=True)) # 6 always 0?
                # data.append(int.from_bytes(line[26:28], byteorder="big", signed=True)) # 7 ?
                # data.append(int.from_bytes(line[28:30], byteorder="big", signed=True)) # 8 ?
                # data.append(int.from_bytes(line[30:32], byteorder="big", signed=True)) # 9 ?
                # data.append(int.from_bytes(line[32:34], byteorder="big", signed=True)) # 10 col 7 + col 9 -2
                # data.append(int.from_bytes(line[24:27], byteorder="big", signed=True)) # 11 ?
                # data.append(int.from_bytes(line[27:28], byteorder="big", signed=True)) # 12 ?
                # data.append(int.from_bytes(line[28:30], byteorder="big", signed=True)) # 13 ?
                # data.append(int.from_bytes(line[30:32], byteorder="big", signed=True)) # 14 ?
                # data.append(int.from_bytes(line[32:34], byteorder="big", signed=True)) # 15 ?
                # data.append(int.from_bytes(line[34:38], byteorder="big", signed=True)) # 16 ?
                # data.append(int.from_bytes(line[38:42], byteorder="big", signed=True)) # 17  col 7 + col 9 -2
                # data.append(int.from_bytes(line[42:47], byteorder="big", signed=True)) #18 ?
                # data.append(int.from_bytes(line[47:58], byteorder="big", signed=True)) #19 ?
                # data.append(int.from_bytes(line[58:63], byteorder="big", signed=True)) #20 same as two before?
                # data.append(int.from_bytes(line[63:78], byteorder="big", signed=True)) #21 ?

                cbj.append((team_black, team_white))
                i += 1
            
        return cbj
    
   
   
# % read headers

    # cbh_path = files["cbh"]
    def read_cbh(cbh_path):
        """
        Function that reads a .cbh header file.

        Parameters
        ----------
        cbh_path : str
            The file path.

        Returns
        -------
        cbh : list
            A list of dictionaries containing the header information.

        """
        with open(cbh_path, "rb") as cbh_file:
            cbh_text = cbh_file.read()
         
        cbh = []
        # the header ends at byte 46, a header entry has a length of 46
        i = 0
        while i*46 + 46 + 46 <= len(cbh_text):
            line = cbh_text[46 + i*46 : 46 + i*46 + 46]
    
            # this is the first bit of this byte
            deleted = line[0] >> 7
            # and this is the last but one, so move one to the right
            # and ignore the other bits (& 1 is and with 00000001)
            guiding = (line[0] >> 1) & 1
               
            cbg_pos = int.from_bytes(line[1:5], byteorder="big", signed=False)
            cba_pos = int.from_bytes(line[5:9], byteorder="big", signed=False)
            player_w_nr = int.from_bytes(
                line[9:12], byteorder="big", signed=False)
            player_b_nr = int.from_bytes(
                line[12:15], byteorder="big", signed=False)
            tournament_nr = int.from_bytes(
                line[15:18], byteorder="big", signed=False)
            annotator_nr = int.from_bytes(
                line[18:21], byteorder="big", signed=False)
            source_nr = int.from_bytes(
                line[21:24], byteorder="big", signed=False)
            try:
                team_white = headers2[i][0]
            except IndexError:
                team_white = 0
            try:
                team_black = headers2[i][1]
            except IndexError:
                team_black = 0
            if team_white > 0:
                team_white = teams[team_white]
            else:
                team_white = "?"
            if team_black > 0:
                team_black = teams[team_black]
            else:
                team_black = "?"            
            
            annotator = "?" if annotator_nr > len(annotators) - 1 else annotators[annotator_nr]     
            source = "?" if source_nr > len(sources) - 1 else sources[source_nr]                   
            
            tournament = {"Event": "?"} if tournament_nr > len(tournaments) - 1 else tournaments[tournament_nr]                   
               
            #print(tournament_nr, player_w_nr, player_b_nr)
            date = line[24:27]
            date = bin(int(date.hex(), 16))[2:]
            if date[15:21] and date[11:15] and date[:11]:
                day = int(date[15:21], 2)
                month = int(date[11:15], 2)
                year = int(date[:11], 2)    
                date = f"{year:04d}.{month:02d}.{day:02d}".replace(".00", ".??")
            else:
                date = "????.??.??"
            
            result = result_map[line[27]]
            
            round_nr = line[29]
            sub_round_nr = line[30]
            white_elo = int.from_bytes(
                line[31:33], byteorder="big", signed=False)
            black_elo = int.from_bytes(
                line[33:35], byteorder="big", signed=False)
            if not white_elo:
                white_elo = "?"
            if not black_elo:
                black_elo = "?"
                
            # first 7 bits: 0-99
            eco = int.from_bytes(line[35:37], byteorder="big", signed=False)
            eco_sub = eco & 127
            eco_standard = eco >> 7
            if eco_standard:
                # A00 is 1, A99 100, E99 500
                eco_standard -= 1
                eco_standard = chr(int(eco_standard / 100) + 65) + str(eco_standard)[-2:].zfill(2)
                if eco_sub:
                    eco_standard += "/" + str(eco_sub)
            else:
                eco_standard = "?"     
            
            if round_nr:
                round_str = str(round_nr) if not sub_round_nr else str(round_nr) + "." + str(sub_round_nr)
            else:
                round_str = ""
            
            cbh.append({    
                "White": (players[player_w_nr] if player_w_nr < len(players) else "?"),
                "Black": (players[player_b_nr] if player_b_nr < len(players) else "?"),
                **tournament,
                "Round": round_str,
                "Date": date,
                "Result": result,
                "WhiteElo": white_elo,
                "BlackElo": black_elo,
                "ECO": eco_standard,
                "Source": source,
                "Annotator": annotator,
                "WhiteTeam": team_white,
                "BlackTeam": team_black,
                "deleted": deleted,
                "guiding": guiding,
                "cbg_pos": cbg_pos,
                "cba_pos": cba_pos,
                })
            i += 1
        return cbh
    
    # % putting everything together

    players = read_cbp(files["cbp"])
    tournaments = read_cbt(files["cbt"])
    headers2 = read_cbj(files["cbj"])
    teams = read_cbe(files["cbe"])
    sources = read_cbs(files["cbs"])
    annotators = read_cbc(files["cbc"])
    headers = read_cbh(files["cbh"])  
    
    if len(headers) > max_games:
        print(f"File {file_cbh} contains {len(headers)} games, which is more than max_games ({max_games}) games. Stopping conversion.")
        return
    
# % read annotations

    # cba_path = files["cba"]
    def read_cba(cba_path):
        """
        Function that reads a .cba source file.

        Parameters
        ----------
        cba_path : str
            The file path.

        Returns
        -------
        cba : dict
            A dictionary with the game_id as index lists of dictionaries with the comments inside.

        """
        global special_chars
        with open(cba_path, "rb") as cba_file:
            cba_text = cba_file.read()
        cba = {}

        len_ann_start = 14
        
        # sometimes the comments start earlier, so determine start from cbh file
        commented_games = [c for i, c in enumerate(headers) if headers[i]["cba_pos"] != 0]
        # empty comments
        if len(commented_games) == 0:
            return cba
        
        i = commented_games[0]["cba_pos"]
        
        j = i + len_ann_start
        while j <= len(cba_text):
            line_h = cba_text[i: j]
            game_id = int.from_bytes(
                line_h[0:3], byteorder="big", signed=False)
            cba[game_id] = []
            len_ann = int.from_bytes(
                line_h[10:14], byteorder="big", signed=False)
            
            # empty string, happens rarely
            if len_ann == 0:
                i = j
                j = i + len_ann_start                   
            
            line_a = cba_text[j : i+len_ann]

            # next iteration
            i = i + len_ann
            j = i + len_ann_start

            while (line_a):
                pos_game = int.from_bytes(
                    line_a[0:3], byteorder="big", signed=False)
                ann_code = line_a[3:4]
                len_text = int.from_bytes(
                    line_a[4:6], byteorder="big", signed=False)
                # rare empty comments - no idea why they exist, but skipping...
                if len_text == 0:
                    break
                                    
                # only comments before and after moves
                if ann_code in [b"\x02", b"\x03", b"\x82"]:
                    ann_type = ann_types[ann_code]
                    
                    if ann_code in [b"\x02", b"\x82"]: # text comment
                        ann_text = decode_text(line_a[8:len_text])

                        if ann_code == b"\x82":
                            ann_text = f"({ann_text}...)"
                        
                    elif ann_code == b"\x03": #symbols
                        symbol = move_comments[line_a[6:7]]
                        ann_text = symbol # (symbol + " " + evaluation).strip()
                        if len_text > 7:
                            evaluation = move_comments[line_a[7:8]] if line_a[7:8] in move_comments else ''
                            ann_text = evaluation
                        if len_text > 8:
                            prefix = special_chars[line_a[8:9]] if line_a[8:9] in special_chars else ''
                            ann_text = "(" + prefix + "...)"
                    
                    # short comments
                    ann_text = ann_text.replace("[%t Shrt] ", "")
                    cba[game_id].append({
                        "halfmove": pos_game,
                        "type": ann_type,
                        "text": ann_text,
                        })
                    
                line_a = line_a[len_text:]
            if not cba[game_id]:
                del cba[game_id]
        return cba    
                                    
# % annotations
    annotations = read_cba(files["cba"])

# % read moves
    
    # King
    # ----
    king = {
    0x49: (0, +1),
    0x39: (+1, +1),
    0xD8: (+1, 0),
    0x5D: (+1, +7),
    0xC2: (0, +7),
    0xB1: (+7, +7),
    0xB2: (+7, 0),
    0x47: (+7, +1),
    }
    
    castling = {        
    0x76: "O-O",
    0xB5: "O-O-O",
    }
    # First queen
    # -----------
    q1 = {
    0xA5: (0, +1),
    0xB8: (0, +2),
    0xCB: (0, +3),
    0x53: (0, +4),
    0x7F: (0, +5),
    0x6B: (0, +6),
    0x8D: (0, +7),
    0x79: (+1, 0),
    0xBE: (+2, 0),
    0xEB: (+3, 0),
    0x21: (+4, 0),
    0x99: (+5, 0),
    0xD2: (+6, 0),
    0x57: (+7, 0),
    0x4D: (+1, +1),
    0xB4: (+2, +2),
    0xBF: (+3, +3),
    0x62: (+4, +4),
    0xBD: (+5, +5),
    0x24: (+6, +6),
    0x96: (+7, +7),
    0xA7: (+1, +7),
    0x48: (+2, +6),
    0x28: (+3, +5),
    0x6E: (+4, +4),
    0x2F: (+5, +3),
    0x5A: (+6, +2),
    0x18: (+7, +1),
    }
    
    # First rook (a1/a8 at start)
    # ----------
    r1 = {
    0x4E: (0, +1),
    0xF8: (0, +2),
    0x43: (0, +3),
    0xD7: (0, +4),
    0x63: (0, +5),
    0x9C: (0, +6),
    0xE6: (0, +7),
    0x2E: (+1, 0),
    0xC6: (+2, 0),
    0x26: (+3, 0),
    0x88: (+4, 0),
    0x30: (+5, 0),
    0x61: (+6, 0),
    0x6F: (+7, 0),
    }
    
    # Second rook (h1/h8 at start)
    # -----------
    r2 = {
    0x14: (0, +1),
    0xA9: (0, +2),
    0x68: (0, +3),
    0xEE: (0, +4),
    0xFB: (0, +5),
    0x77: (0, +6),
    0xE2: (0, +7),
    0xA6: (+1, 0),
    0x05: (+2, 0),
    0x8B: (+3, 0),
    0xA1: (+4, 0),
    0x98: (+5, 0),
    0x32: (+6, 0),
    0x52: (+7, 0),
    }
    
    # First bishop (c1/c8 at start)
    # ------------
    b1 = {
    0x02: (+1, +1),
    0x97: (+2, +2),
    0xE1: (+3, +3),
    0x41: (+4, +4),
    0xC3: (+5, +5),
    0x7C: (+6, +6),
    0xE4: (+7, +7),
    0x06: (+1, +7),
    0xB7: (+2, +6),
    0x55: (+3, +5),
    0xD9: (+4, +4),
    0x2C: (+5, +3),
    0xAE: (+6, +2),
    0x37: (+7, +1),
    }
    
    # Second bishop (f1/f8 at start)
    # -------------
    b2 = {        
    0xF6: (+1, +1),
    0x3F: (+2, +2),
    0x08: (+3, +3),
    0x93: (+4, +4),
    0x73: (+5, +5),
    0x5E: (+6, +6),
    0x78: (+7, +7),
    0x35: (+1, +7),
    0xF2: (+2, +6),
    0x6D: (+3, +5),
    0x71: (+4, +4),
    0xA2: (+5, +3),
    0xF3: (+6, +2),
    0x16: (+7, +1),
    }
    
    # First knight (b1/b8 at start)
    # ------------
    n1 = {
    0x58: (+2, +1),
    0x3D: (+1, +2),
    0xFA: (-1, +2),
    0xE9: (-2, +1),
    0xBA: (-2, -1),
    0xD4: (-1, -2),
    0xDD: (+1, -2),
    0x4A: (+2, -1),
    }
    
    # Second knight (g1/g8 at start)
    # -------------
    n2 =  {
    0xC4: (+2, +1),
    0x0E: (+1, +2),
    0xFE: (-1, +2),
    0x5F: (-2, +1),
    0x75: (-2, -1),
    0x07: (-1, -2),
    0x89: (+1, -2),
    0x34: (+2, -1),
    }
    
    # a2/a7-pawn
    # ------------
    p1 = {
    0x2D: (0, 1),
    0xC1: (0, 2),
    0x8E: (1, 1),
    0xF5: (-1, 1),
    }
    
    # b2/b7-pawn
    # ------------
    p2 = {
    0x64: (0, 1),
    0x17: (0, 2),
    0x70: (1, 1),
    0xA4: (-1, 1),
    }
    
    # c2/c7-pawn
    # ------------
    p3 = {
    0x7B: (0, 1),
    0xDA: (0, 2),
    0xE0: (1, 1),
    0x85: (-1, 1),
    }
    # d2/d7-pawn
    # ------------
    p4 = {
    0xC5: (0, 1),
    0x0B: (0, 2),
    0x90: (1, 1),
    0xF9: (-1, 1),
    }
    
    # e2/e7-pawn
    # ------------
    p5 = {
    0x84: (0, 1),
    0xFF: (0, 2),
    0x15: (1, 1),
    0x36: (-1, 1),
    }
    # f2/f7-pawn
    # ------------
    p6 = {
    0x09: (0, 1),
    0x9E: (0, 2),
    0x7D: (1, 1),
    0xDE: (-1, 1),
    }
    
    # g2/g7-pawn
    # ------------
    p7 = {
    0xBB: (0, 1),
    0xDF: (0, 2),
    0xBC: (1, 1),
    0x3A: (-1, 1),
    }
    
    # h2/h7-pawn
    # ------------
    p8 = {
    0x12: (0, 1),
    0x33: (0, 2),
    0x13: (1, 1),
    0x19: (-1, 1),
    }
    # Second queen
    # ------------
    q2 = {
    0xE5: (0, +1),
    0x94: (0, +2),
    0x50: (0, +3),
    0x11: (0, +4),
    0xEA: (0, +5),
    0x31: (0, +6),
    0x01: (0, +7),
    0x5C: (+1, 0),
    0x95: (+2, 0),
    0xCA: (+3, 0),
    0xD3: (+4, 0),
    0x1D: (+5, 0),
    0x7E: (+6, 0),
    0xEF: (+7, 0),
    0x44: (+1, +1),
    0x80: (+2, +2),
    0xA0: (+3, +3),
    0x1F: (+4, +4),
    0x83: (+5, +5),
    0x00: (+6, +6),
    0x4B: (+7, +7),
    0x67: (+1, +7),
    0x20: (+2, +6),
    0x5B: (+3, +5),
    0x2A: (+4, +4),
    0x92: (+5, +3),
    0xB6: (+6, +2),
    0x60: (+7, +1),
    }
    
    # Third queen
    # -----------
    q3 = {
    0x1A: (0, +1),
    0x42: (0, +2),
    0x0F: (0, +3),
    0x0D: (0, +4),
    0xB0: (0, +5),
    0xD1: (0, +6),
    0x23: (0, +7),
    0xF0: (+1, 0),
    0x7A: (+2, 0),
    0x54: (+3, 0),
    0x4F: (+4, 0),
    0xF4: (+5, 0),
    0xA8: (+6, 0),
    0x72: (+7, 0),
    0xE7: (+1, +1),
    0x40: (+2, +2),
    0x38: (+3, +3),
    0x59: (+4, +4),
    0x87: (+5, +5),
    0xE8: (+6, +6),
    0x6C: (+7, +7),
    0x86: (+1, +7),
    0x04: (+2, +6),
    0xF1: (+3, +5),
    0x8C: (+4, +4),
    0xCE: (+5, +3),
    0x6A: (+6, +2),
    0xDB: (+7, +1),
    }
    
    # Third rook
    # ----------
    r3 = {
    0x81: (0, +1),
    0x82: (0, +2),
    0x9A: (0, +3),
    0x1B: (0, +4),
    0x9D: (0, +5),
    0x0A: (0, +6),
    0x2B: (0, +7),
    0x8F: (+1, 0),
    0xCD: (+2, 0),
    0xED: (+3, 0),
    0x10: (+4, 0),
    0x74: (+5, 0),
    0x69: (+6, 0),
    0xD6: (+7, 0),
    }
    
    # Third bishop
    # ------------
    b3 = {
    0x51: (+1, +1),
    0xB9: (+2, +2),
    0x45: (+3, +3),
    0x3B: (+4, +4),
    0x56: (+5, +5),
    0x91: (+6, +6),
    0xFD: (+7, +7),
    0xAB: (+1, +7),
    0x66: (+2, +6),
    0x3E: (+3, +5),
    0x46: (+4, +4),
    0xB3: (+5, +3),
    0xFC: (+6, +2),
    0xC8: (+7, +1),
    }
    
    # Third knight
    # ------------
    n3 = {
    0x9B: (+2, +1),
    0xC0: (+1, +2),
    0xE3: (-1, +2),
    0xA3: (-2, +1),
    0xAC: (-2, -1),
    0xC9: (-1, -2),
    0xEC: (+1, -2),
    0x27: (+2, -1),
    }
        
    move_codes = {
        "k": king,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "n1": n1,
        "n2": n2,
        "n3": n3,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6": p6,
        "p7": p7,
        "p8": p8,
        "castling": castling,
        }
   
    code2move = {code: (piece, change) for piece, codes in move_codes.items() for code, change in codes.items()}
    
    promotion_codes = ["Q", "R", "B", "N"]
    
    setup_codes = {
    "10001": "K",
    "10010": "Q",
    "10011": "N",
    "10100": "B",
    "10101": "R",
    "10110": "P",
    "11001": "k",
    "11010": "q",
    "11011": "n",
    "11100": "b",
    "11101": "r",
    "11110": "p",
    }
    
    moveNumberLookup = [
    	0xa2, 0x95, 0x43, 0xf5, 0xc1, 0x3d, 0x4a, 0x6c,	#   0 -   7
    	0x53, 0x83, 0xcc, 0x7c, 0xff, 0xae, 0x68, 0xad,	#   8 -  15
    	0xd1, 0x92, 0x8b, 0x8d, 0x35, 0x81, 0x5e, 0x74,	#  16 -  23
    	0x26, 0x8e, 0xab, 0xca, 0xfd, 0x9a, 0xf3, 0xa0,	#  24 -  31
    	0xa5, 0x15, 0xfc, 0xb1, 0x1e, 0xed, 0x30, 0xea,	#  32 -  39
    	0x22, 0xeb, 0xa7, 0xcd, 0x4e, 0x6f, 0x2e, 0x24,	#  40 -  47
    	0x32, 0x94, 0x41, 0x8c, 0x6e, 0x58, 0x82, 0x50,	#  48 -  55
    	0xbb, 0x02, 0x8a, 0xd8, 0xfa, 0x60, 0xde, 0x52,	#  56 -  63
    	0xba, 0x46, 0xac, 0x29, 0x9d, 0xd7, 0xdf, 0x08,	#  64 -  71
    	0x21, 0x01, 0x66, 0xa3, 0xf1, 0x19, 0x27, 0xb5,	#  72 -  79
    	0x91, 0xd5, 0x42, 0x0e, 0xb4, 0x4c, 0xd9, 0x18,	#  80 -  87
    	0x5f, 0xbc, 0x25, 0xa6, 0x96, 0x04, 0x56, 0x6a,	#  88 -  95
    	0xaa, 0x33, 0x1c, 0x2b, 0x73, 0xf0, 0xdd, 0xa4,	#  96 - 103
    	0x37, 0xd3, 0xc5, 0x10, 0xbf, 0x5a, 0x23, 0x34,	# 104 - 111
    	0x75, 0x5b, 0xb8, 0x55, 0xd2, 0x6b, 0x09, 0x3a,	# 112 - 119
    	0x57, 0x12, 0xb3, 0x77, 0x48, 0x85, 0x9b, 0x0f,	# 120 - 127
    	0x9e, 0xc7, 0xc8, 0xa1, 0x7f, 0x7a, 0xc0, 0xbd,	# 128 - 135
    	0x31, 0x6d, 0xf6, 0x3e, 0xc3, 0x11, 0x71, 0xce,	# 136 - 143
    	0x7d, 0xda, 0xa8, 0x54, 0x90, 0x97, 0x1f, 0x44,	# 144 - 151
    	0x40, 0x16, 0xc9, 0xe3, 0x2c, 0xcb, 0x84, 0xec,	# 152 - 159
    	0x9f, 0x3f, 0x5c, 0xe6, 0x76, 0x0b, 0x3c, 0x20,	# 160 - 167
    	0xb7, 0x36, 0x00, 0xdc, 0xe7, 0xf9, 0x4f, 0xf7,	# 168 - 175
    	0xaf, 0x06, 0x07, 0xe0, 0x1a, 0x0a, 0xa9, 0x4b,	# 176 - 183
    	0x0c, 0xd6, 0x63, 0x87, 0x89, 0x1d, 0x13, 0x1b,	# 184 - 191
    	0xe4, 0x70, 0x05, 0x47, 0x67, 0x7b, 0x2f, 0xee,	# 192 - 199
    	0xe2, 0xe8, 0x98, 0x0d, 0xef, 0xcf, 0xc4, 0xf4,	# 200 - 207
    	0xfb, 0xb0, 0x17, 0x99, 0x64, 0xf2, 0xd4, 0x2a,	# 208 - 215
    	0x03, 0x4d, 0x78, 0xc6, 0xfe, 0x65, 0x86, 0x88,	# 216 - 223
    	0x79, 0x45, 0x3b, 0xe5, 0x49, 0x8f, 0x2d, 0xb9,	# 224 - 231
    	0xbe, 0x62, 0x93, 0x14, 0xe9, 0xd0, 0x38, 0x9c,	# 232 - 239
    	0xb2, 0xc2, 0x59, 0x5d, 0xb6, 0x72, 0x51, 0xf8,	# 240 - 247
    	0x28, 0x7e, 0x61, 0x39, 0xe1, 0xdb, 0x69, 0x80,	# 248 - 255
    ]
    
    def pos2square(array):
        """
        Convert square array to string.

        Parameters
        ----------
        array : np.array
            The square as array (0-7, 0-7).

        Returns
        -------
        str
            The square in the usual notation (like e4).

        """
        file = chr(97 + array[0])
        rank = str(array[1] + 1)
        # assert array[1] < 8 and array[1] >= 0 and array[0] < 8 and array[0] >= 0, "invalid square"
        return file + rank

# setup start position
    def setup_position():
        """
        Generate the start position boards.

        Returns
        -------
        pos : TYPE
            DESCRIPTION.
        posi : TYPE
            DESCRIPTION.

        """
        posi = {
            "R1": [0, 0],
            "R2": [7, 0],
            "N1": [1, 0],
            "N2": [6, 0],
            "B1": [2, 0],
            "B2": [5, 0],
            "Q1": [3, 0],
            "K": [4, 0],    
            "turn_w": True,
            }
        for i in range(8):
            posi[f"P{i + 1}"] = [i, 1]
            
        pos = {}
        for key, value in posi.items():
            if type(value) != bool:
                pos[key.lower()] = np.array([value[0], 7 - value[1]])
                pos[key] = np.array(value)
            else:
                pos[key] = value
        
        # allow searching for square
        posi = {str(value): key for key, value in pos.items() if key != "turn_w"}
        return pos, posi
    
    # "29" = multiple byte move to follow

# %%
    
    # cbg_path = files["cbg"]
    def read_cbg(cbg_path):
        # %%
        with open(cbg_path, "rb") as cbg_file:
            cbg_text = cbg_file.read()
        cbg = []
    
        len_game_start = 4
                 
        for game_no in range(len(headers)):
            
            game_info = headers[game_no].copy()
            
            # skip guiding texts
            if game_info["guiding"]:
                continue
            
            game_anno = annotations[game_no + 1] if game_no + 1 in annotations else []
            combined_anno = {}
            for anno in game_anno:
                if (anno["halfmove"] + 1) not in combined_anno:
                    combined_anno[anno["halfmove"] + 1] = "{" + anno["text"] + "}"
                else:
                    combined_anno[anno["halfmove"] + 1] += "{" + anno["text"] + "}"
            game_anno = combined_anno
                        
            position, positioni = setup_position()
            
            # sometimes there's some extra stuff, so determine i from cbh file
            i = game_info['cbg_pos']
            
            j = i + len_game_start
                
            game_header = cbg_text[i : j]
            bistr = bin(int.from_bytes(game_header, byteorder="big", signed=False))[2:].zfill(32)
                        
            setup_pos = bool(int(bistr[1]))
            bytes_game = int(bistr[2 : 32], 2) - 4
            
            i = j
                            
            if setup_pos:
                position = {}
                # the fifth bit from the end is 0: shift by 4 so bit -5 is last,
                # then use & 1 to set all other bits to 0
                position["turn_w"] = (((cbg_text[i + 1]) >> 4) & 1) == 0
                
                # & 15 are the last four bits (other bits become 0)                
                enpassant_square =  chr(96 + (cbg_text[i + 1] & 15)) \
                    + ("6" if position["turn_w"] else "3")
                if enpassant_square[0] == "`":
                    enpassant_square = "-"
                castling_b = cbg_text[i + 2]
                # & 1 is the last, & 2 the lat but one, & 4 the last but two bit and so on                
                position["O-O-O"] = bool(castling_b & 1)
                position["O-O"] = bool(castling_b & 2)
                position["o-o-o"] = bool(castling_b & 4)
                position["o-o"] = bool(castling_b & 8)
                                
                pos_string = str(bin(int(cbg_text[i + 4: i + 28].hex(), 16)))[2:].zfill(24*8)
                temp_pos = []
                while pos_string:
                    if pos_string[0] == "0":
                        temp_pos.append(None)
                        pos_string = pos_string[1:]
                    else:
                        temp_pos.append(setup_codes[pos_string[:5]])
                        pos_string = pos_string[5:]
                for k, p in enumerate(temp_pos):
                    if p:
                        for l in range(1, 9):
                            coordinates = np.array([int(k / 8), k % 8])
                            piece = f"{p}{l}" if p.lower() != "k" else p
                            if piece not in position:
                                position[piece] = coordinates
                                break
                
                # generate FEN
                fen = ""
                for y in np.arange(7, -1, -1):
                    for x in range(8):
                        found = False
                        for piece, square in position.items():
                            if str(square) == f"[{x} {y}]":
                                fen += piece[0]
                                found = True
                                break
                        if not found: fen += " "
                    fen += "/"
                for x in np.arange(8, 0, -1):
                    fen = fen.replace(" " * x, str(x))
                fen = fen.strip("/")
                fen += " " + ("w" if position["turn_w"] else "b")
                fen += " "
                fen += ("K" if position["O-O"] else "")
                fen += ("Q" if position["O-O-O"] else "")
                fen += ("k" if position["o-o"] else "")
                fen += ("q" if position["o-o-o"] else "")
                
                for castle in ["o-o-o", "o-o", "O-O", "O-O-O"]:
                    del position[castle]
                
                # field to piece for faster processing            
                positioni = {str(value): key for key, value in position.items() if key != "turn_w"}
                
                if fen[-1] == " ":
                    fen += "-"
                fen += " " + enpassant_square
                fen += " 0 1"
                game_info["FEN"] = fen
                        
                i = i + 28
                bytes_game -= 28
                
            k = 0

            previous = []
            
            moves = []
            if k in game_anno:
                moves.append(game_anno[k])
            # i = 30
            

            while k < bytes_game - 1:
                
                # subtract the current move number
                m = (cbg_text[i] - k) % 256
                # print(m)
                i += 1
                turn = position["turn_w"]
                capture = False
               
                # variations and game end
                if m == 220:
                    bytes_game -= 1
                    previous.append((position.copy(), positioni.copy()))
                    moves.append("(")
                elif m == 12:
                    bytes_game -= 1
                    # not game end
                    if len(previous):
                        position, positioni = previous.pop()
                        moves.append(")")
                    else:
                        # this happens rarely, no idea why
                        i -= 1
                        break
                        # raise Exception("unexpected game end")
                    
                # multi-byte move
                elif m == 41:

                    position["turn_w"] = not position["turn_w"]
                    
                    # these are two bytes, so get the number and shift it 8 bits and append the next one
                    # to get the two bytes as number
                    multi_byte = (moveNumberLookup[(cbg_text[i] - k) % 256] << 8) + \
                    (moveNumberLookup[(cbg_text[i + 1] - k) % 256])

                    i += 2
                    k += 1
                    bytes_game -= 2
                    
                    # the first 6 bits (& 63) are the first square; a1 = 0, a2 = 1, ..., h8 = 63
                    # you need to decode the byte first as done above using moveNumberLookup!
                    # >> 3 is the same as dividing by 8 without remainder
                    first_square = np.array([(multi_byte & 63) >> 3, multi_byte % 8])
                    # the next six the second square
                    second_square = np.array([((multi_byte >> 6) & 63) >> 3, (multi_byte >> 6) % 8])
                    
                    # delete captured piece
                    if str(second_square) in positioni:
                        del position[positioni[str(second_square)]]
                        capture = True

                    # find the moving piece
                    piece = positioni[str(first_square)]
                    del positioni[str(first_square)]
 
                    # pawn promotion
                    if piece[0].lower() == "p" and (second_square[1] == 0 or second_square[1] == 7):
                        # the next 3 bits are the promotion code
                        promotion = promotion_codes[(multi_byte >> 12) & 3]
                        if not turn:
                            promotion = promotion.lower()
                        del position[piece]
                        # create new piece
                        for n in range(1, 9):
                            if f"{promotion}{n}" not in position:
                                position[f"{promotion}{n}"] = second_square
                                positioni[str(second_square)] = f"{promotion}{n}"
                                break
                        moves.append(pos2square(first_square) + pos2square(second_square) + promotion.upper())
                    else:
                        position[piece] = second_square
                        positioni[str(second_square)] = piece
                        
                        moves.append(piece[0].strip("Pp").upper() + pos2square(first_square) + pos2square(second_square))
                        
                    if k in game_anno:
                        moves.append(game_anno[k])
                                # null move
                elif m == 170:
                    k += 1
                    position["turn_w"] = not position["turn_w"]
                    moves.append("--")
                    if k in game_anno:
                        moves.append(game_anno[k])                           
                    
                # dummy, counts as byte but does nothing
                elif m == 159:
                    bytes_game -= 1
                    
                else:
                    p, change = code2move[m]

                    if p == "castling":
                        k += 1
                        position["turn_w"] = not position["turn_w"]
                        # king white or black
                        p_k = "K" if turn else "k"
                        rank = (7 + turn) % 8
                        if change == "O-O-O":                                   
                            p_r = "R1" if turn else "r1"
                            del positioni[str(position[p_r])]
                            del positioni[str(position[p_k])]
                            position[p_k] = np.array([2, rank])
                            position[p_r] = np.array([3, rank])
                            positioni[f"[2 {rank}]"] = p_k
                            positioni[f"[3 {rank}]"] = p_r

                        elif change == "O-O":                                   
                            p_r = "R2" if turn else "r2"
                            if p_r not in position:
                                p_r = p_r[0] + "1"
                            del positioni[str(position[p_r])]
                            del positioni[str(position[p_k])]
                            position[p_k] = np.array([6, rank])
                            position[p_r] = np.array([5, rank])
                            positioni[f"[6 {rank}]"] = p_k
                            positioni[f"[5 {rank}]"] = p_r

                        moves.append(change)
                        if k in game_anno:
                            moves.append(game_anno[k])
                        
                    else:
                        k += 1
                        position["turn_w"] = not position["turn_w"]
                        p = p.upper() if turn else p
                       
                        if p[0].lower() != "p":
                            # remove captured piece
                            move_target = (position[p] + change) % 8
                            if str(move_target) in positioni:
                               del position[positioni[str(move_target)]]
                               capture = True
                                
                            now = pos2square(position[p])
                            then = pos2square(move_target)
                            piece = p[0].upper()
                            moves.append(piece + now + then)
                            if k in game_anno:
                                moves.append(game_anno[k])
                            del positioni[str(position[p])]
                            position[p] = move_target
                            positioni[str(move_target)] = p
                     
                        else:
                            # the direction is reversed for black
                            change = change if turn else np.negative(change)
                            move_target = (position[p] + change) % 8
                            # capturing
                            if change[0] != 0:
                                found = False
                                if str(move_target) in positioni:
                                    capture = True
                                    del position[positioni[str(move_target)]]
                                    found = True
                                    
                                # en passant: capturing empty square, pawn is next to pawn
                                if not found:
                                    move_target_ep = (position[p] + (change[0], 0)) % 8
                                    capture = True
                                    del position[positioni[str(move_target_ep)]]
                                    del positioni[str(move_target_ep)]
                                    found = True
                                    
                         
                            now = pos2square(position[p])
                            then = pos2square(move_target)
                            moves.append(now + then)
                            if k in game_anno:
                                moves.append(game_anno[k])
                                
                            del positioni[str(position[p])]
                            position[p] = move_target
                            positioni[str(move_target)] = p
                
                if capture:
                    # "promote" pieces to lowest number: if first rook is captured, R2 becomes R1
                    keys = sorted(position.keys()).copy()
                    for key in keys:
                        value = position[key]
                        if key[-1].isnumeric() and key[0].lower() != "p" and key[-1] != "1":
                            if not (key[0] + str(int(key[1]) - 1)) in position:
                                position[key[0] + str(int(key[1]) - 1)] = value
                                del position[key]
                                positioni[str(value)] = key[0] + str(int(key[1]) - 1)


            # add comments to moves
            moves = "\f\v".join(moves).replace("\f\v{", "{")
            # remove brackets from NAGs
            for nag in move_comments.values():
                moves = moves.replace("{" + nag + "}", nag)
            moves = moves.split("\f\v")
            
            game_moves = {f"\f\f{i}\f\f": [] for i in range( moves.count("(") + 1)}
            level = 0
            var = 0
            temp = []

            # determine variation structure
            opened = 0
            brackets = []
            closed_brackets = []
           
            for bracket in [m for m in moves if m in "()"]:
                if bracket == "(":
                    opened += 1
                    brackets.append(opened)
                elif bracket == ")":
                    closed_brackets.append(brackets.pop())
  
            for mve_nr in range(len(moves)):
                m = moves[mve_nr]
                if m == "(":
                    var += 1
                    m = f"\f\f{var}\f\f"
                    
                    # fix variation begin after variation - alternative line
                    if moves[mve_nr - 1] == ")":
                        game_moves[f"\f\f{level}\f\f"] += [m]
                        continue                      
                    
                    temp.append(m)   
                    continue                                 
                
                elif m == ")":
                    level = closed_brackets.pop(0)               
                    continue
                
                game_moves[f"\f\f{level}\f\f"].append(m)
                
                if temp:
                    game_moves[f"\f\f{level}\f\f"] += temp
                    temp = []
            
            mainline = " ".join(game_moves["\f\f0\f\f"])
            
            # put variations starting at the beginning of a variation behind the start of the variation
            for lev, variation in game_moves.items():
                if variation[0].startswith("\f\f"):
                    mainline = mainline.replace(lev, lev + " " + variation.pop(0))
                mainline = mainline.replace(lev, " ( " + " ".join(variation) + " ) ")
                
            # remove outer brackets and add space to comment begin
            san_moves = mainline.strip("() ").replace("{", " {").replace("  ", " ")
            
            # these tags are required in that very order.
            roster = ["Event", "Site", "Date", "Round", "White", "Black", "Result"]

            pgn_header = "\n".join([
                f'[{key} "{game_info[key]}"]' if key in game_info else f'[{key} "??"]' for key in roster
                ])
            pgn_header += "\n" + "\n".join([
                f'[{key} "{value}"]' for key, value in game_info.items()
                    if key[0].isupper()
                    and key not in roster
                ])
            
            game_string = pgn_header + "\n\n" + san_moves
            
            # ok, this is slow and stupid, I could determine the proper move strings before and check validity.
            # If anyone wants to add a "find the proper short notation" algorithm above,
            # feel free to submit a merge request :-)
            game = chess.pgn.read_game(io.StringIO(game_string))
            cbg.append(game)

        return cbg
    
    games = read_cbg(files["cbg"])
    if output == "pychess":
        return games
    else:
        pgn = "\n\n".join([str(game) for game in games])
        
        if output == "text":
            return pgn
        
        with open(file_cbh.rsplit(".", 1)[0] + ".pgn", "w") as file:
            file.write(pgn)
    

# %%
