import numpy as np

from flightSim.aircraft import AltCmd, HdgCmd, SpdCmd

CmdCount = [20, 7, 20, 9, 20, 7]


def int_2_cmd(now: int, idx: list):
    # alt cmd
    time_idx, cmd_idx = int(np.argmax(idx[:CmdCount[0]]))*10, idx[CmdCount[0]:]
    alt_idx, cmd_idx = int(np.argmax(cmd_idx[:CmdCount[1]])), cmd_idx[CmdCount[1]:]
    alt_cmd = AltCmd(delta=(alt_idx - 3) * 300.0, assignTime=now+time_idx)
    print('{:>3d} {:>2d}'.format(time_idx, alt_idx), end='\t')

    # hdg cmd
    time_idx, cmd_idx = int(np.argmax(cmd_idx[:CmdCount[2]]))*10, cmd_idx[CmdCount[2]:]
    hdg_idx, cmd_idx = int(np.argmax(cmd_idx[:CmdCount[3]])), cmd_idx[CmdCount[3]:]
    hdg_cmd = HdgCmd(delta=(hdg_idx - 4) * 15, assignTime=now+time_idx)
    print('{:>3d} {:>2d}'.format(time_idx, hdg_idx), end='\t')

    # spd cmd
    time_idx, cmd_idx = int(np.argmax(cmd_idx[:CmdCount[4]]))*10, cmd_idx[CmdCount[4]:]
    spd_idx = int(np.argmax(cmd_idx))
    spd_cmd = SpdCmd(delta=(spd_idx - 3) * 10, assignTime=now+time_idx)
    print('{:>3d} {:>2d}'.format(time_idx, spd_idx), end='\t')

    return [alt_cmd, hdg_cmd, spd_cmd]


def rew_for_cmd(conflict_acs, cmd_info):
    rewards = []
    for ac in conflict_acs:
        [alt_cmd, hdg_cmd, spd_cmd] = cmd_info[ac]
        rew_alt = 0.3 - abs(alt_cmd.delta) / 3000.0
        rew_hdg = 0.4 - abs(hdg_cmd.delta) / 150.0
        rew_spd = 0.3 - abs(spd_cmd.delta) / 100.0
        rewards.append(rew_alt+rew_spd+rew_hdg)
    return rewards
