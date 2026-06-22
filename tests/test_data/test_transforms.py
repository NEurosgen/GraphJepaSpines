import pytest
import torch
from torch_geometric.data import Data
from src.data_utils.transforms import (
    MaskData,
    BuildGeometricGraph,
    NormNoEps,
    LocalPos,
    FeatureChoice,
    GenNormalize,
    create_mask_collate_fn,
    Rotate,
    build_augments,
)


@pytest.fixture
def dummy_data():
    x = torch.randn(10, 5)            # 10 узлов, 5 признаков
    pos = torch.randn(10, 3)          # 3D позиции
    edge_index = torch.randint(0, 10, (2, 20))
    edge_attr = torch.randn(20, 2)    # 2-мерный edge_attr — должен быть перетёрт канонизацией
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def line_graph():
    """h01-подобный граф: только x и pos (узлы на прямой с шагом 1.0), без рёбер."""
    x = torch.ones(5, 3)
    pos = torch.tensor([[float(i), 0.0, 0.0] for i in range(5)])
    return Data(x=x, pos=pos)


def test_mask_data(dummy_data):
    context, target = MaskData(mask_ratio=0.3)(dummy_data)
    assert isinstance(context, Data) and isinstance(target, Data)
    assert context.num_nodes + target.num_nodes == dummy_data.num_nodes


def test_build_geometric_graph_from_positions(line_graph):
    out = BuildGeometricGraph(r=1.5)(line_graph)
    # ребра построены из pos, edge_attr = дистанции формы [E, 1]
    assert out.edge_index.size(0) == 2
    assert out.edge_attr.dim() == 2 and out.edge_attr.size(1) == 1
    # при шаге 1.0 и r=1.5 соединяются только соседи (dist=1.0), не через одного (2.0)
    assert torch.allclose(out.edge_attr, torch.ones_like(out.edge_attr), atol=1e-5)
    # 4 неупорядоченные пары соседей -> 8 направленных рёбер
    assert out.edge_index.size(1) == 8


def test_build_geometric_graph_overwrites_existing(dummy_data):
    # вход с edge_attr размерности 2 -> после канонизации всегда [E, 1] (дистанции)
    out = BuildGeometricGraph(r=1.0)(dummy_data)
    assert out.edge_attr.size(1) == 1


def test_build_geometric_graph_on_edgeless_input(line_graph):
    # на входе нет рёбер (h01) — структура восстанавливается из pos
    assert line_graph.edge_index is None
    out = BuildGeometricGraph(r=1.5)(line_graph)
    assert out.edge_index.numel() > 0


def test_norm_no_eps(dummy_data):
    transform = NormNoEps(mean=torch.zeros(5), std=torch.ones(5))
    # значения <= eps не нормализуются (маска по |x| > eps)
    dummy_data.x[0, :] = 1e-8
    out = transform(dummy_data)
    assert torch.allclose(out.x[0, :], torch.full((5,), 1e-8))
    assert out.x.shape == (10, 5)


def test_local_pos(dummy_data):
    out = LocalPos()(dummy_data)
    assert out.pos.shape == (10, 3)
    assert torch.allclose(out.pos.mean(dim=0), torch.zeros(3), atol=1e-5)


def test_feature_choice(dummy_data):
    out = FeatureChoice(feature=[0, 2, 4])(dummy_data)
    assert out.x.shape == (10, 3)


def test_gen_normalize(dummy_data):
    out = GenNormalize(transforms=[LocalPos()])(dummy_data)
    assert isinstance(out, Data)

    ctx, tgt = GenNormalize(
        transforms=[LocalPos()], mask_transform=MaskData(mask_ratio=0.5)
    )(dummy_data)
    assert isinstance(ctx, Data) and isinstance(tgt, Data)


def test_rotate_changes_pos_but_preserves_distances(dummy_data):
    pos_before = dummy_data.pos.clone()
    # попарные дистанции — инвариант жёсткого поворота
    d_before = torch.cdist(pos_before, pos_before)

    out = Rotate(angle_deg=90, axis="z")(dummy_data)
    assert not torch.allclose(out.pos, pos_before)             # позиции изменились
    d_after = torch.cdist(out.pos, out.pos)
    assert torch.allclose(d_before, d_after, atol=1e-5)        # дистанции сохранены


def test_rotate_identity(dummy_data):
    pos_before = dummy_data.pos.clone()
    out = Rotate(angle_deg=0)(dummy_data)
    assert torch.allclose(out.pos, pos_before)


def test_rotate_90_in_xy_plane():
    # вокруг z: (x, y) -> (-y, x); z неизменна
    data = Data(x=torch.ones(1, 2), pos=torch.tensor([[1.0, 0.0, 5.0]]))
    out = Rotate(angle_deg=90, axis="z")(data)
    assert torch.allclose(out.pos, torch.tensor([[0.0, 1.0, 5.0]]), atol=1e-5)


def test_build_augments():
    cfg = {"rotation_angles": [0, 90, 180, 270], "rotation_axis": "z"}
    augs = build_augments(cfg)
    assert augs is not None and len(augs) == 4
    assert all(isinstance(a, Rotate) for a in augs)
    # выключено
    assert build_augments({"rotation_angles": []}) is None
    assert build_augments({}) is None


def test_collate_with_rotation_augments(line_graph):
    """Список augments -> по одному представлению на угол, каждое со своей маской."""
    canonical = GenNormalize([LocalPos(), BuildGeometricGraph(r=1.5)], mask_transform=None)
    base = [canonical(line_graph.clone())]  # 1 базовый граф

    dyn = GenNormalize([], mask_transform=MaskData(mask_ratio=0.4))
    augs = build_augments({"rotation_angles": [0, 90, 180, 270]})
    collate = create_mask_collate_fn(dyn, num_views=1, augments=augs)

    ctx_batch, tgt_batch = collate(base)
    # 1 граф * 4 угла = 4 представления
    assert int(ctx_batch.batch.max()) + 1 == 4


def test_collate_num_views(line_graph):
    """num_views размножает (context, target) пары: N сэмплов * V представлений."""
    canonical = GenNormalize([LocalPos(), BuildGeometricGraph(r=1.5)], mask_transform=None)
    base = [canonical(line_graph.clone()), canonical(line_graph.clone())]

    dyn = GenNormalize([], mask_transform=MaskData(mask_ratio=0.4))
    collate = create_mask_collate_fn(dyn, num_views=3)

    ctx_batch, tgt_batch = collate(base)
    # 2 сэмпла * 3 представления = 6 графов
    assert int(ctx_batch.batch.max()) + 1 == 6
    assert int(tgt_batch.batch.max()) + 1 == 6
