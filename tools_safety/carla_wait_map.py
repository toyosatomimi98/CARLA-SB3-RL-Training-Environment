"""Load Town10HD_Opt, then poll until the map becomes queryable."""
import time
import carla


def try_map(client, tag):
    try:
        w = client.get_world()
        m = w.get_map()
        sp = m.get_spawn_points()
        print(f"[{tag}] OK map={m.name} spawns={len(sp)}")
        return w, m, sp
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] err: {type(e).__name__}: {str(e)[:70]}")
        return None, None, None


def main() -> None:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60.0)
    try:
        client.load_world("Town10HD_Opt")
        print("load_world issued")
    except Exception as e:  # noqa: BLE001
        print("load_world err:", type(e).__name__, str(e)[:80])
    for i in range(12):
        time.sleep(10)
        w, m, sp = try_map(client, f"t+{10 * (i + 1)}s")
        if w is not None:
            bl = w.get_blueprint_library()
            veh = [b.id for b in bl.filter("vehicle.*")]
            print("   vehicle blueprints:", len(veh), veh[:12])
            break


if __name__ == "__main__":
    main()
