"""
    `voo_evasion` is a Python module which allows to detect and control
    a .évasion box from VOO (Belgium)
    Copyright (C) 2019 Vincent STRAGIER (vincent.stragier@outlook.com)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from functools import partial
import argparse
import json
import requests
import socket
import traceback
import os
import pkg_resources
import sys

KNOWN_PORTS_RFB = [5900]

KEYS_FILE = pkg_resources.resource_string(__name__, "evasion_keys.json")
COMMANDS = json.loads(KEYS_FILE)
COMMANDS_HTTP = COMMANDS["voo tv+"]
COMMANDS_RFB = COMMANDS["legacy"]

COMMANDS_RFB_LS = list(COMMANDS_RFB.keys())
COMMANDS_HTTP_LS = list(COMMANDS_HTTP.keys())


class ManageVerbose:
    """Allows to mask print() to the user."""

    def __init__(self, verbose=True):
        self.verbosity = verbose

    def __enter__(self):
        if not self.verbosity:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.verbosity:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def is_RFB_and_like_VOO_evasion(ip, port, timeout=2, verbose=False):
    """Check the behaviour of the connection mechanism to detect the .evasion
     box(es)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
        try:
            tcp.settimeout(timeout)
            tcp.connect((ip, port))
            serverMSG = tcp.recv(4096)

            if serverMSG[0:3] != b'RFB':  # Not using RFB
                # print(serverMSG)
                return False

            tcp.send(serverMSG)  # Continue the handcheck

            serverMSG = tcp.recv(4096)

            # Security is not the same as on the .evasion box
            if serverMSG != b'\x01\x01':
                # print(serverMSG)
                return False

            tcp.send(b'\x01')
            serverMSG = tcp.recv(4096)

            # ?? Error about security negotiation
            if serverMSG != b'\x00\x00\x00\x00':
                return False

            tcp.send(b'\x01')
            serverMSG = tcp.recv(4096)

            if serverMSG == bytes(24):
                return True
            # print(len(serverMSG))
            return False

        except Exception:
            if verbose:
                print(traceback.format_exc())
            return False


def command_to_set_volume(volume: int) -> list:
    # NOT WORKING (? add a delay ?)
    VOL_DOWN, VOL_UP = 57348, 57347
    cmd = [VOL_DOWN for _ in range(21)]

    for _ in range(volume):
        cmd.append(VOL_UP)

    return cmd


def is_RFB_and_like_VOO_evasion_Pool(
        ip, port=5900, timeout=0.5, verbose=False):
    """Adaptation of "is_RFB_and_like_VOO_evasion()" for Pool."""
    if is_RFB_and_like_VOO_evasion(ip, port, timeout, verbose):
        return ip
    return 0


def scan_RFB(ports: list):
    """Scan the default interface network (all the IPs on the  interface)
    to find .evasion box(es) and return a list of potential boxes."""
    import netifaces
    import ipaddress
    from multiprocessing import Pool

    # Find default network interface
    default_iface = netifaces.gateways()['default'][netifaces.AF_INET]
    # Addresses de l'interface
    addrs = netifaces.ifaddresses(default_iface[1])[netifaces.AF_INET]

    ls = []

    for addr in addrs:
        for port in ports:
            print(port)
            # Extract and compute subnet mask
            mask = format(int(ipaddress.ip_address(addr["netmask"])), "32b")
            print(f"{mask = }")

            cnt_1 = mask.count("1")
            if cnt_1 > 0 and mask == f"{'1' * cnt_1}{'0' * (32 - cnt_1)}":
                mask = f"{cnt_1}"
            else:
                print('Error, invalid mask address.')
                mask = 'invalid mask'

            ip_base = ipaddress.ip_network(
                f"{addr['broadcast'].replace('255', '0')}/{mask}")

            # Base IP address
            print(f"{ip_base = }")

            addresses = [str(address) for address in ip_base]

            # Configure pool (scaling depend of the number of CPU)
            n = os.cpu_count() * 25
            n_max = 60  # Seems to be a limit in Windows...
            if n > n_max:
                n = n_max
            print(f"Pool size (max={n_max}): {n}")

            # Scan all the addresses with the help of a pool
            scan_function = partial(
                is_RFB_and_like_VOO_evasion_Pool,
                port=port
            )
            with Pool(n) as p:
                ls_evasion = p.map(scan_function, addresses)

            # Clean the results
            ls_evasion = [
                (element, port) for element in ls_evasion if element != 0]
            ls.extend(ls_evasion)
            # print(ls)

    print(f"Scan completed ({len(ls)} address{'es' if len(ls) > 1 else ''}).")
    return ls


def display_commands(commands_dict: dict, display: bool = True):
    """Display the list of valid know command (name and value)."""
    commands = list(commands_dict.keys())
    commands.sort()

    commands_list = (
        f"{command} = {commands_dict[command]}" for command in commands)

    if display:
        print('\n'.join(commands_list))
    return commands_list


def is_valid_command(commands_dict: dict, temp_command: str):
    """Check the command validity (non case sensitive)."""
    if temp_command.upper() in list(commands_dict.keys()):
        return True
    elif True:
        try:
            cmd = int(temp_command)
            if cmd in commands_dict.values():
                return True
            else:
                return False
        except Exception:
            return False


def convert_command_to_value(commands_dict: dict, temp_command: str):
    """Convert a command (name or value to value)."""
    if is_valid_command(commands_dict, temp_command):
        try:
            return commands_dict[temp_command]
        except Exception:  # Maybe it is already a value.
            return int(temp_command)


def convert_command(commands_dict: dict, temp_command: str):
    """Convert a command (value to name or name to value)."""
    try:
        return commands_dict[temp_command]
    except Exception:
        keys = []
        if keys is None:
            print("NO KEY FOUND")
        for key, val in commands_dict.items():
            if val == int(temp_command):
                keys.append(key)
        if keys:
            temp_str = ' or '
            return temp_str.join(keys)

        raise NameError(
            'Did not find the corresponding command name for this value')


def type_port(a_string: str):
    """Check the "type" in the parser for the port."""
    try:
        p = int(a_string)
        if p >= 0 and p <= 65535:
            return a_string
        else:
            raise argparse.ArgumentTypeError(
                "Value should be an integer between 0 and 65535 included.")
    except Exception:
        raise argparse.ArgumentTypeError(
            "Value should be an integer between 0 and 65535 included.")


def type_command(a_string: str):
    """Check the "type" in the parser for the command."""
    if (is_valid_command(COMMANDS_HTTP, a_string)
            or is_valid_command(COMMANDS_RFB, a_string)):
        return a_string
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid command ('{a_string}'), use '-lc' to list the valid "
            "commands")


def display_security_type(sec_type: int, display: bool = True) -> None:
    """Display the security type of the RFB connection."""
    switcher = {0: "Invalid", 1: "NONE", 2: "VNC Authentication"}
    security_type = switcher.get(sec_type, "Not defined by IETF")
    if display:
        print(f' security type: {sec_type} ({security_type})')
    return security_type


def gen_packet_from_cmd(cmd: str) -> bytes:
    """
    Generate a bytes array with the command to send (KEYDOWN_KEY, KEYUP_KEY).
    """
    return (b'\x04\x01\x00\x00' + (cmd).to_bytes(4, byteorder='big')
            + b'\x04\x00\x00\x00' + (cmd).to_bytes(4, byteorder='big'))


def channel_to_command(commands_dict: dict, channel: str):
    """Convert a channel number in a sequence of command."""
    cmd_ls = []
    try:
        channel_str = str(abs(channel))
    except Exception:
        print(traceback.format_exc())
        exit(1)

    for figure in channel_str:
        cmd_ls.append(convert_command_to_value(
            commands_dict, f'REMOTE_{figure}'))
    cmd_ls.append(convert_command_to_value(commands_dict, 'OK'))
    return cmd_ls


def send_cmd(ip, port, cmd, protocol="RFB", timeout=None):
    """Send the command to the defined address."""
    try:
        if protocol == "RFB":
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as vnc:
                print(
                    f"1. Initialize connection to:\n IP: {ip}\n Port: {port}")

                if (timeout and (
                        isinstance(timeout, float)
                        or isinstance(timeout, int))):
                    vnc.settimeout(timeout)
                vnc.connect((ip, 5900))
                vnc_ver = vnc.recv(12)
                print(f"Receive RFB protocol version: {vnc_ver}")
                vnc.send(vnc_ver)
                print(f"Send back RFB protocol version: {vnc_ver}")
                print("2. Receive security")
                nb_sec_types = ord(vnc.recv(1))
                # print(nb_sec_types.decode())
                print(f"Nb security types: {nb_sec_types}")

                for _ in range(nb_sec_types):
                    sec_type = ord(vnc.recv(1))
                    display_security_type(sec_type)

                ClientInit = b'\x01'
                # Send 1 byte
                print(f"3. Send ClientInit message: {ClientInit}")
                # 0 -> Non exclusive connection, 1 -> exclusive connection
                vnc.send(ClientInit)

                print(f"Receive ServerInit: {vnc.recv(4096)}")

                ClientEncodage = b'\x01'
                # Send 1 byte
                print(f"4. Send Client to server message: {ClientEncodage}")
                vnc.send(ClientEncodage)

                print(f"Receive from ServerClient: {vnc.recv(4096)}")
                if isinstance(cmd, str):
                    print(
                        f"5. Send command '{cmd}': {gen_packet_from_cmd(cmd)}"
                    )
                    vnc.send(gen_packet_from_cmd(cmd))
                elif isinstance(cmd, list):
                    if len(cmd) > 1:
                        print("5. Send multiple commands:")
                    else:
                        print("5. ", end="")

                    for c in cmd:
                        print(
                            f"\tSend command '{c}': {gen_packet_from_cmd(c)}")
                        vnc.send(gen_packet_from_cmd(c))
                else:
                    raise NameError('In function send_cmd(), "cmd" should be'
                                    ' a string or a list of string.')
                return True, None

        elif protocol == "HTTP":
            print(f"To:\n IP: {ip} PORT: {port}")
            if isinstance(cmd, str):
                print(f"Send command '{cmd}': {cmd}")
                data = {'code': cmd}
                r = requests.post(
                    url=f"http://{ip}:{port}/apps/mzcast/run/rcu", json=data)
                if r.status_code != 202:
                    raise NameError('In function send_cmd(), send error')
            elif isinstance(cmd, list):
                if len(cmd) > 1:
                    print("Send multiple commands:")
                else:
                    print("", end="")

                for c in cmd:
                    data = {'code': c}
                    r = requests.post(
                        url=f"http://{ip}:{port}/apps/mzcast/run/rcu",
                        json=data)
                    if r.status_code != 202:
                        raise requests.RequestException(
                            f'In function send_cmd(), error {r.status_code}')
            else:
                raise NameError('In function send_cmd(), "cmd" should be a '
                                'string or a list of string.')
            return True, None

    except Exception as e:
        print(traceback.format_exc())
        return False, e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", default=False,
                        help="increase output verbosity",
                        action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-http", action="store_true", default=True,
                      help="use the HTTP mode for VOO TV+ .evasion boxes "
                      "(default).")
    mode.add_argument("-rfb", action="store_true", default=False,
                      help="use RFB for legacy .evasion boxes (default is to "
                      "use RFB).")
    parser.add_argument("-f", "--find", default=False,
                        help="return a list of potential .evasion boxes.",
                        action="store_true")
    parser.add_argument("-s", "--status", default=False,
                        help="return 'success' if the command has been send "
                             "else it return 'fail'.",
                        action="store_true")
    parser.add_argument("-a", "--address", type=str,
                        help="IP address of the .evasion box")
    parser.add_argument("-p", "--port", type=type_port, default=-1,
                        help="port of the .evasion box, default is 5900 (in "
                        "mode RFB) [optional], note that VOO TV+ uses 38520 "
                        "(default in mode HTTP)")
    parser.add_argument("-c", "--command", type=type_command, nargs='+',
                        help="command to send to the .evasion box (the "
                             "command is checked), name of the command and "
                             "value are accepted")
    parser.add_argument("-ch", "--channel", type=int,
                        help="send the command to the .evasion box to change "
                             "the channel (must be an integer)")

    # # NOT WORKING
    # parser.add_argument("-vol", "--volume", type=int,
    #                     help="send the command to the .evasion box to change"
    #                     " the volume (must be an integer [0-20])")
    # # Not implemented
    # parser.add_argument("-rc", "--raw_command", type=int,
    #                     help="raw command which will be send to the .evasion"
    #                     " box (will be send as it is without check), must be"
    #                     " an integer")
    parser.add_argument("-cv", "--convert_command",
                        type=type_command, nargs='+',
                        help="convert a valid command from name to value or "
                             "from value to name")
    parser.add_argument("-lc", "--list_commands",
                        help="display the list of known commands",
                        action="store_true")

    args = parser.parse_args()
    args.http = not args.rfb
    mode = "HTTP" if args.http else "RFB"
    if args.port == -1:
        args.port = 38520 if args.http else 5900

    if args.verbose:
        print("Python 3 based .evasion box API")
        print("Verbosity turned on.\n")
        print("Arguments:\n")
        for arg, value in vars(args).items():
            print(f"\t'{arg}': {value}")
        print()

    if args.list_commands:
        if args.verbose:
            print("Display the list of valid know commands for the .evasion "
                  "box:\n")
        print("RFB based command:")
        display_commands(COMMANDS_RFB)
        print("\nHTTP based command:")
        display_commands(COMMANDS_HTTP)

    if args.convert_command:
        with ManageVerbose(args.verbose):
            print(f"Command(s) or value(s) for {mode} mode: ")
        for cmd in args.convert_command:
            with ManageVerbose(args.verbose):
                print(f"\t{cmd}: ", end='')
            print(convert_command(
                COMMANDS_HTTP if mode == "HTTP" else COMMANDS_RFB,
                cmd.upper()))

    if args.find:
        print("Start scanning network (this is a CPU intensive task, which "
              "needs the 'netifaces' module and only works with legacy "
              ".evasion boxes which use the RBF protocol instead of HTTP):")
        with ManageVerbose(args.verbose):
            evasion = scan_RFB(KNOWN_PORTS_RFB)

        if len(evasion) > 0:
            if len(evasion) == 1:
                print("Potential .evasion box:")
            else:
                print("Potential .evasion boxes:")
            for box in evasion:
                print(f"IP: {box[0]}, port: {box[1]}")
        else:
            print("No box have been found.")

    # Set channel
    if args.address and args.channel:
        try:
            with ManageVerbose(args.verbose):
                cmd_ls = channel_to_command(
                    COMMANDS_HTTP if mode == "HTTP" else COMMANDS_RFB,
                    args.channel)
                print(cmd_ls)
                result, error = send_cmd(args.address, args.port, cmd_ls, mode)

            if result and (args.status or args.verbose):
                print('Success')
            elif args.status or args.verbose:
                print('Fail')
                if args.verbose:
                    print(error)

        except Exception:
            print(traceback.format_exc())

    """
    NOT WORKING
    if args.address and args.volume:
        try:
            with ManageVerbose(args.verbose):
                cmd_ls = command_to_set_volume(args.volume)
                print(cmd_ls)
                result, error = send_cmd(args.address, args.port, cmd_ls)

            if result and (args.status or args.verbose):
                print('Success')
            elif args.status or args.verbose:
                print('Fail')
                if args.verbose:
                    print(error)

        except Exception:
            print(traceback.format_exc())
    """

    # Send command to the box
    if args.address and args.command:
        try:
            cmd_ls = []
            for cmd in args.command:
                cmd_ls.append(convert_command_to_value(
                    COMMANDS_HTTP if mode == "HTTP" else COMMANDS_RFB,
                    cmd.upper()))

            with ManageVerbose(args.verbose):
                print(cmd_ls)
                result, error = send_cmd(
                    args.address, args.port, cmd_ls, mode)

            if result and (args.status or args.verbose):
                print('Success')
            elif args.status or args.verbose:
                print('Fail')
                if args.verbose:
                    print(error)

        except Exception:
            print(traceback.format_exc())

    """
    # Create a socket to be used a client

    socket.setdefaulttimeout(10)

    timeout = socket.getdefaulttimeout()

    print(f"System has default timeout of {timeout} for create_connection")

    with socket.create_connection(("192.168.0.15",5900)) as s:
        print("connected")
        bytes2Send = str.encode("Hello server system!")
        # s.sendall(bytes2Send)
        # Receive the data

        data = s.recv(1024)

        print(data)
    """


if __name__ == "__main__":
    import warnings
    mod = "voo_evasion"
    warnings.warn(
        f"use 'python -m {mod}', not 'python -m {mod}.{mod}'",
        DeprecationWarning)
    main()
