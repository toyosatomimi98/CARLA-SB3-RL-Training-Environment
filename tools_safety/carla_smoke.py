"""Smoke test: connect to CARLA 0.10.0 and probe the APIs this repo relies on."""
import time
import carla


def probe_map(client, world, label):
    print(f"\n=== map probe: {label} ===")
    try:
        m = world.get_map()
        print("  map name:", m.name)
        sp = m.get_spawn_points()
        print("  spawn points:", len(sp))
        topo = m.get_topology()
        print("  topology edges:", len(topo))
        if topo:
            w0, w1 = topo[0]
            print("  first edge:", w0.transform.location, "->", w1.transform.location)
        od = m.to_opendrive()
        print("  opendrive chars:", len(od))
    except Exception as e:  # noqa: BLE001
        print("  FAIL:", type(e).__name__, e)
    try:
        print("  get_world() after load:", client.get_world().get_map().name)
    except Exception as e:  # noqa: BLE001
        print("  get_world() FAIL:", type(e).__name__, e)


def main() -> None:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    print("server:", client.get_server_version(), "client:", client.get_client_version())
    maps = client.get_available_maps()
    print("available maps:", len(maps))
    for m in maps[:8]:
        print("   ", m)

    try:
        world = client.get_world()
    except Exception as e:  # noqa: BLE001
        print("get_world error:", type(e).__name__, e)
        world = client.load_world("Town10HD_Opt")
    probe_map(client, world, "current")
    probe_map(client, client.load_world("Town10HD_Opt"), "Town10HD_Opt")

    try:
        print("current map:", world.get_map().name)
    except Exception as e:  # noqa: BLE001
        print("map name error:", type(e).__name__, e)

    print("\n-- probing APIs used by the repo --")
    checks = [
        ("load_world('Town02')", lambda: client.load_world("Town02")),
        ("bp vehicle.tesla.model3", lambda: world.get_blueprint_library().find("vehicle.tesla.model3")),
        ("bp sensor.camera.rgb", lambda: world.get_blueprint_library().find("sensor.camera.rgb")),
        ("bp sensor.camera.semantic_segmentation",
         lambda: world.get_blueprint_library().find("sensor.camera.semantic_segmentation")),
        ("bp sensor.other.collision", lambda: world.get_blueprint_library().find("sensor.other.collision")),
        ("bp sensor.other.lane_invasion", lambda: world.get_blueprint_library().find("sensor.other.lane_invasion")),
        ("bp sensor.lidar.ray_cast", lambda: world.get_blueprint_library().find("sensor.lidar.ray_cast")),
        ("spawn_points", lambda: world.get_map().get_spawn_points()),
        ("Settings", lambda: world.get_settings()),
        ("VehicleControl", lambda: carla.VehicleControl()),
        ("ColorConverter.Raw", lambda: carla.ColorConverter.Raw),
        ("ColorConverter.CityScapesPalette", lambda: carla.ColorConverter.CityScapesPalette),
    ]
    for name, fn in checks:
        try:
            r = fn()
            extra = ""
            if name == "spawn_points":
                extra = f" n={len(r)} first={r[0].location}"
            print(f"  OK   {name} {extra}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {name}: {type(e).__name__}: {e}")

    # waypoint API + transform
    try:
        wp = world.get_map().get_waypoint(world.get_map().get_spawn_points()[0].location)
        print("  OK   get_waypoint ->", wp.transform.location, "road_id", wp.road_id)
    except Exception as e:  # noqa: BLE001
        print("  FAIL get_waypoint:", e)

    # spawn + tick + destroy
    try:
        bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("color", bp.get_attribute("color").recommended_values[0])
        v = world.spawn_actor(bp, world.get_map().get_spawn_points()[0])
        print("  OK   spawn vehicle:", v.type_id)
        time.sleep(1.0)
        v.set_simulate_physics(False)
        v.set_transform(world.get_map().get_spawn_points()[1])
        v.set_simulate_physics(True)
        print("  OK   set_simulate_physics / set_transform")
        print("  OK   velocity:", v.get_velocity())
        v.destroy()
        print("  OK   destroy")
    except Exception as e:  # noqa: BLE001
        print("  FAIL spawn/tick:", type(e).__name__, e)


if __name__ == "__main__":
    main()
