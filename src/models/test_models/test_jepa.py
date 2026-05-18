import pytest
import torch
from src.models.jepa import CrossAttentionPredictor, LeJEPA, sigreg
from src.models.encoder import GraphGinEncoder


class SimpleGraph:
    """Minimal graph-like object with the attributes LeJEPA.forward() expects."""
    def __init__(self, num_nodes: int, in_channels: int, seed: int = 0):
        g = torch.Generator()
        g.manual_seed(seed)
        self.x = torch.randn(num_nodes, in_channels, generator=g)
        self.pos = torch.randn(num_nodes, 3, generator=g)
        self.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.edge_attr = torch.empty(0)


@pytest.fixture
def hidden_dim():
    return 16


@pytest.fixture
def predictor_inputs(hidden_dim):
    pos_dim = 3
    torch.manual_seed(0)
    context_emb = torch.randn(10, hidden_dim)
    context_pos = torch.randn(10, pos_dim)
    target_pos = torch.randn(5, pos_dim)
    return hidden_dim, pos_dim, context_emb, context_pos, target_pos


@pytest.fixture
def small_lejepa():
    in_channels, hidden_dim = 5, 16
    encoder = GraphGinEncoder(in_channels=in_channels, out_channels=hidden_dim, num_layers=1)
    predictor = CrossAttentionPredictor(hidden_dim=hidden_dim, pos_dim=3)
    model = LeJEPA(encoder=encoder, predictor=predictor, lambd=0.1, num_slices=32)
    context = SimpleGraph(num_nodes=10, in_channels=in_channels, seed=0)
    target = SimpleGraph(num_nodes=5, in_channels=in_channels, seed=1)
    return model, context, target


# ── CrossAttentionPredictor ──────────────────────────────────────────────────

def test_predictor_output_shape(predictor_inputs):
    hidden_dim, pos_dim, context_emb, context_pos, target_pos = predictor_inputs
    predictor = CrossAttentionPredictor(hidden_dim=hidden_dim, pos_dim=pos_dim)

    pred = predictor(context_emb=context_emb, context_pos=context_pos, target_pos=target_pos)

    assert pred.shape == (5, hidden_dim)


def test_predictor_no_nan(predictor_inputs):
    hidden_dim, pos_dim, context_emb, context_pos, target_pos = predictor_inputs
    predictor = CrossAttentionPredictor(hidden_dim=hidden_dim, pos_dim=pos_dim)

    pred = predictor(context_emb=context_emb, context_pos=context_pos, target_pos=target_pos)

    assert not torch.isnan(pred).any()
    assert not torch.isinf(pred).any()


def test_predictor_gradient_flow(predictor_inputs):
    """mask_token, pos_embed и MLP должны получать градиенты."""
    hidden_dim, pos_dim, context_emb, context_pos, target_pos = predictor_inputs
    predictor = CrossAttentionPredictor(hidden_dim=hidden_dim, pos_dim=pos_dim)

    pred = predictor(context_emb=context_emb, context_pos=context_pos, target_pos=target_pos)
    pred.sum().backward()

    assert predictor.mask_token.grad is not None, "mask_token не получил градиент"
    assert predictor.pos_embed.weight.grad is not None, "pos_embed.weight не получил градиент"
    assert predictor.mlp[0].weight.grad is not None, "MLP не получил градиент"


def test_predictor_cross_attention_uses_context(predictor_inputs):
    """Выход должен меняться при изменении context_emb — cross-attention работает."""
    hidden_dim, pos_dim, context_emb, context_pos, target_pos = predictor_inputs
    predictor = CrossAttentionPredictor(hidden_dim=hidden_dim, pos_dim=pos_dim)
    predictor.eval()

    with torch.no_grad():
        pred_original = predictor(context_emb=context_emb, context_pos=context_pos, target_pos=target_pos)
        pred_zero_ctx = predictor(context_emb=torch.zeros_like(context_emb), context_pos=context_pos, target_pos=target_pos)

    assert not torch.allclose(pred_original, pred_zero_ctx), \
        "Выход не зависит от context_emb — cross-attention не работает"


def test_predictor_residual_uses_target_query(predictor_inputs):
    """Выход должен меняться при изменении target_pos — residual path активен."""
    hidden_dim, pos_dim, context_emb, context_pos, target_pos = predictor_inputs
    predictor = CrossAttentionPredictor(hidden_dim=hidden_dim, pos_dim=pos_dim)
    predictor.eval()

    target_pos_shifted = target_pos + 10.0

    with torch.no_grad():
        pred1 = predictor(context_emb=context_emb, context_pos=context_pos, target_pos=target_pos)
        pred2 = predictor(context_emb=context_emb, context_pos=context_pos, target_pos=target_pos_shifted)

    assert not torch.allclose(pred1, pred2), \
        "Выход не зависит от target_pos — residual с mask_token не работает"


def test_predictor_variable_target_size(hidden_dim):
    """Предиктор должен работать с любым числом target-узлов."""
    pos_dim = 3
    predictor = CrossAttentionPredictor(hidden_dim=hidden_dim, pos_dim=pos_dim)
    context_emb = torch.randn(8, hidden_dim)
    context_pos = torch.randn(8, pos_dim)

    for num_targets in [1, 3, 20]:
        target_pos = torch.randn(num_targets, pos_dim)
        pred = predictor(context_emb=context_emb, context_pos=context_pos, target_pos=target_pos)
        assert pred.shape == (num_targets, hidden_dim)


# ── LeJEPA ───────────────────────────────────────────────────────────────────

def test_lejepa_loss_is_scalar(small_lejepa):
    model, context, target = small_lejepa
    loss = model(context, target)

    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_lejepa_loss_positive(small_lejepa):
    """MSE ≥ 0 и sigreg ≥ 0 → суммарный loss ≥ 0."""
    model, context, target = small_lejepa
    with torch.no_grad():
        loss = model(context, target)
    assert loss.item() >= 0.0


def test_lejepa_backward(small_lejepa):
    """Энкодер (кроме замороженного proj) и предиктор должны получать градиенты."""
    model, context, target = small_lejepa
    loss = model(context, target)
    loss.backward()

    assert model.encoder.proj.weight.grad is None, "proj должен быть заморожен"
    assert model.encoder.layers[0].model.nn[0].weight.grad is not None, "GIN MLP не получил градиент"
    assert model.predictor.pos_embed.weight.grad is not None, "predictor не получил градиент"
    assert model.predictor.mask_token.grad is not None, "mask_token не получил градиент"


def test_lejepa_lambda_zero_equals_mse(small_lejepa):
    """При lambd=0 loss должен совпадать с чистым MSE(pred, target_enc).

    Модель переводится в eval(), чтобы Dropout в predictor был детерминирован
    и оба вычисления (явное и внутри model.forward) дали одинаковый MSE.
    """
    model, context, target = small_lejepa
    model.lambd = 0.0
    model.eval()

    with torch.no_grad():
        context_enc = model.encoder(context.x, context.edge_index, context.edge_attr)
        target_enc = model.encoder(target.x, target.edge_index, target.edge_attr)
        pred = model.predictor(
            context_emb=context_enc, context_pos=context.pos, target_pos=target.pos
        )
        expected = model.loss_fn(pred, target_enc)
        actual = model(context, target)

    assert torch.allclose(actual, expected, atol=1e-5), \
        f"При lambd=0 ожидался чистый MSE={expected.item():.6f}, получили {actual.item():.6f}"


def test_lejepa_lambda_affects_loss(small_lejepa):
    """Разные значения lambd должны давать разный loss."""
    model, context, target = small_lejepa

    with torch.no_grad():
        model.lambd = 0.0
        loss_no_reg = model(context, target).item()

        model.lambd = 1.0
        loss_full_reg = model(context, target).item()

    assert abs(loss_no_reg - loss_full_reg) > 1e-6, \
        "lambd не влияет на loss — возможна ошибка в формуле"


def test_lejepa_encode_method(small_lejepa):
    """Метод encode() должен возвращать node-level эмбеддинги нужной формы."""
    model, context, _ = small_lejepa
    out = model.encode(context.x, context.edge_index, context.edge_attr)
    assert out.shape == (context.x.shape[0], model.encoder.out_channels)


# ── sigreg ───────────────────────────────────────────────────────────────────

def test_sigreg_output_shape():
    x = torch.randn(10, 16)
    reg = sigreg(x, num_slices=64)

    assert reg.dim() == 1
    assert reg.size(0) == 64


def test_sigreg_no_nan():
    x = torch.randn(32, 64)
    reg = sigreg(x, num_slices=128)

    assert not torch.isnan(reg).any()
    assert not torch.isinf(reg).any()


def test_sigreg_is_real():
    """sigreg использует комплексную арифметику внутри, но возвращает вещественный тензор."""
    x = torch.randn(20, 32)
    reg = sigreg(x, num_slices=64)

    assert reg.is_floating_point()
    assert not reg.is_complex()


def test_sigreg_nonnegative():
    """Регуляризация основана на квадратах — значения не могут быть отрицательными."""
    x = torch.randn(16, 32)
    reg = sigreg(x, num_slices=64)

    assert (reg >= 0).all()


def test_sigreg_gaussian_less_than_uniform():
    """Гауссовский вход должен давать меньшую регуляризацию, чем равномерный.

    sigreg штрафует за отклонение от Гауссовского распределения:
    Gaussian → малый штраф, Uniform → большой штраф.
    Абсолютный порог не фиксируем, т.к. sigreg масштабируется на N.
    """
    torch.manual_seed(42)
    x_gauss = torch.randn(512, 32)
    x_uniform = torch.rand(512, 32) * 6 - 3  # равномерно в [-3, 3]

    reg_gauss = sigreg(x_gauss, num_slices=64).mean().item()
    reg_uniform = sigreg(x_uniform, num_slices=64).mean().item()

    assert reg_gauss < reg_uniform, (
        f"Гауссовский вход должен давать меньший sigreg, чем равномерный: "
        f"gauss={reg_gauss:.4f}, uniform={reg_uniform:.4f}"
    )
