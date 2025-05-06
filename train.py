# ✅ train_final.py -- 符合新版F-VCD训练规范（完整版）

import os
import time
import json
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt
from model import *
from dataset_npy_loader import Dataset
from utils import write3d, normalize_percentile
from config import config
from tensorlayer.layers import InputLayer
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.TRAIN.device)

# ================= 图像与训练参数 ====================
img_size = config.img_setting.img_size
n_num = config.img_setting.Nnum
sr_factor = config.img_setting.sr_factor
n_slices = config.img_setting.n_slices
ReScale_factor = config.img_setting.ReScale_factor
channels_interp = config.channels_interp
normalize_mode = config.preprocess.normalize_mode

batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

sample_ratio = config.TRAIN.sample_ratio
shuffle_for_epoch = config.TRAIN.shuffle_for_epoch

checkpoint_dir = config.TRAIN.ckpt_dir
log_dir = config.TRAIN.log_dir
test_saving_dir = config.TRAIN.test_saving_path
plot_test_loss_dir = os.path.join(test_saving_dir, 'test_loss_plt')
test_hr_dir = os.path.join(test_saving_dir, 'HR_View')
test_lf_dir = os.path.join(test_saving_dir, 'LFP')
test_stack_dir = os.path.join(test_saving_dir, 'Target3D')

label = config.label
SR_loss = config.Loss.SR_loss
Recon_loss = config.Loss.Recon_loss
loss_ratio = config.Loss.Ratio

save_hyperparameters = True

# ================ 工具函数 ==================
def weighted_mae_loss(image, reference):
    # 添加数值稳定性的小常数
    epsilon = 1e-8
    # 裁剪以避免极端值
    image = tf.clip_by_value(image, 0.0, 1.0)
    reference = tf.clip_by_value(reference, 0.0, 1.0)
    weight = tf.abs(reference) + epsilon
    diff = tf.abs(image - reference)
    # 裁剪极端差异值
    diff = tf.clip_by_value(diff, 0.0, 1.0)
    loss = tf.reduce_mean(weight * diff)
    return loss

def weighted_mse_loss(image, reference):
    # 添加数值稳定性的小常数
    epsilon = 1e-8
    # 裁剪以避免极端值
    image = tf.clip_by_value(image, 0.0, 1.0)
    reference = tf.clip_by_value(reference, 0.0, 1.0)
    weight = tf.abs(reference) + epsilon
    diff = tf.square(image - reference)
    # 裁剪极端差异值
    diff = tf.clip_by_value(diff, 0.0, 1.0)
    loss = tf.reduce_mean(weight * diff)
    return loss

def edge_loss(image, reference):
    def _gradient(x):
        # 添加数值稳定性的小常数
        epsilon = 1e-7
        x = tf.clip_by_value(x, epsilon, 1.0)
        gx = x[:, :-1, :, :] - x[:, 1:, :, :]
        gy = x[:, :, :-1, :] - x[:, :, 1:, :]
        # 裁剪梯度值避免极端值
        gx = tf.clip_by_value(gx, -0.5, 0.5)
        gy = tf.clip_by_value(gy, -0.5, 0.5)
        return gx, gy

    gx_i, gy_i = _gradient(image)
    gx_r, gy_r = _gradient(reference)

    # 裁剪成统一尺寸
    min_h = tf.minimum(tf.shape(gx_i)[1], tf.shape(gx_r)[1])
    min_w = tf.minimum(tf.shape(gx_i)[2], tf.shape(gx_r)[2])

    gx_i = tf.image.resize_with_crop_or_pad(gx_i, min_h, min_w)
    gx_r = tf.image.resize_with_crop_or_pad(gx_r, min_h, min_w)
    gy_i = tf.image.resize_with_crop_or_pad(gy_i, min_h, min_w)
    gy_r = tf.image.resize_with_crop_or_pad(gy_r, min_h, min_w)

    # 计算差异时添加更多的数值保护
    diff_x = tf.clip_by_value(tf.abs(gx_i - gx_r), 0.0, 0.5) 
    diff_y = tf.clip_by_value(tf.abs(gy_i - gy_r), 0.0, 0.5)
    loss = tf.reduce_mean(diff_x + diff_y)
    return loss

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

class Trainer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.losses = {}
        self.test_loss_plt = []

    def build_graph(self):
        # —— 动态学习率变量 —— 
        with tf.variable_scope('learning_rate'):
            self.learning_rate = tf.Variable(config.TRAIN.lr_init, trainable=False)

        # —— 图像尺寸计算 —— 
        input_size  = np.array([config.img_setting.img_size, config.img_setting.img_size])
        SR_size     = input_size * config.img_setting.sr_factor
        Recon_size  = np.multiply(SR_size, config.img_setting.ReScale_factor)

        # —— 占位符定义 —— 
        self.plchdr_lf         = tf.placeholder('float32', [config.TRAIN.batch_size, *input_size, config.img_setting.Nnum], name='t_LFP')
        self.plchdr_SynView    = tf.placeholder('float32', [config.TRAIN.batch_size, *SR_size,     config.img_setting.Nnum], name='t_SynView')
        self.plchdr_target3d   = tf.placeholder('float32', [config.TRAIN.batch_size, *Recon_size,  config.img_setting.n_slices], name='t_target3d')

        # —— GPU 设备，上下文内保持网络链完整 —— 
        with tf.device(f"/gpu:{config.TRAIN.device}"):
            # 1) 用 InputLayer 包装 LF 输入  
            lf_input = InputLayer(self.plchdr_lf, name='input_LF')

            # 2) 构建 SR 子网  
            self.SR_net = F_Denoise(
                lf_input,
                SR_size,
                angRes=config.img_setting.Nnum,
                sr_factor=config.img_setting.sr_factor,
                reuse=False,
                name=config.net_setting.SR_model,
                channels_interp=config.channels_interp,
                normalize_mode=config.preprocess.normalize_mode
            )
            self.SR_net.print_params(False)

            # 3) 构建 Recon 子网，直接传入 Layer  
            self.Recon_net = F_Recon(
                self.SR_net,
                config.img_setting.n_slices,
                Recon_size,
                reuse=False,
                name=config.net_setting.Recon_model,
                channels_interp=config.channels_interp,
                normalize_mode=config.preprocess.normalize_mode
            )
            self.Recon_net.print_params(False)

        # —— 变量列表 —— 
        SR_vars    = tl.layers.get_variables_with_name(config.net_setting.SR_model,   train_only=True, printable=False)
        Recon_vars = tl.layers.get_variables_with_name(config.net_setting.Recon_model, train_only=True, printable=False)

        # —— 损失计算 —— 
        self.SR_loss    = 0.0
        self.Recon_loss = 0.0

        # 2.1) SR 阶段  
        for key, weight in config.Loss.SR_loss.items():                    
            fn       = globals()[key]
            tmp_loss = tf.clip_by_value(
                fn(image=self.SR_net.outputs, reference=self.plchdr_SynView),
                clip_value_min=1e-8, clip_value_max=1e4
            )
            self.SR_loss += weight * tmp_loss
            self.losses[f"SR_{key}"] = weight * tmp_loss
            tf.summary.scalar(f"SR_{key}", tmp_loss)

        # 2.2) Recon 阶段  
        for key, weight in config.Loss.Recon_loss.items():                 
            fn       = globals()[key]
            tmp_loss = tf.clip_by_value(
                fn(image=self.Recon_net.outputs, reference=self.plchdr_target3d),
                clip_value_min=1e-8, clip_value_max=1e8
            )
            self.Recon_loss += weight * tmp_loss
            self.losses[f"Recon_{key}"] = weight * tmp_loss
            tf.summary.scalar(f"Recon_{key}", tmp_loss)

        # 2.3) 总损失  
        sr_w, recon_w = config.Loss.Ratio                                 
        self.loss     = sr_w * self.SR_loss + recon_w * self.Recon_loss
        self.losses['total_loss'] = self.loss
        tf.summary.scalar('total_loss', self.loss)
        tf.summary.scalar('learning_rate', self.learning_rate)

        # —— Session & TensorBoard —— 
        configProto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        configProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=configProto)

        self.merge_op      = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(config.TRAIN.log_dir, self.sess.graph)

        # —— 优化器（增加梯度裁剪） —— 
        with tf.variable_scope('optimizer'):
            # SR优化器 - 添加梯度裁剪
            sr_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=config.TRAIN.beta1, epsilon=1e-8)
            sr_grads_and_vars = sr_optimizer.compute_gradients(self.SR_loss, var_list=SR_vars)
            sr_clipped_grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) 
                                     for grad, var in sr_grads_and_vars if grad is not None]
            self.SR_optim = sr_optimizer.apply_gradients(sr_clipped_grads_and_vars)

            # Recon优化器 - 添加梯度裁剪
            recon_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=config.TRAIN.beta1, epsilon=1e-8)
            recon_grads_and_vars = recon_optimizer.compute_gradients(self.loss, var_list=SR_vars + Recon_vars)
            recon_clipped_grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) 
                                        for grad, var in recon_grads_and_vars if grad is not None]
            self.Recon_optim = recon_optimizer.apply_gradients(recon_clipped_grads_and_vars)

        # —— 参数打印（可选） —— 
        print("🔍 SR_net 参数总数:", len(SR_vars))
        print("🔍 Recon_net 参数总数:", len(Recon_vars))
        print("🔍 SR_net 参数列表:")
        for v in SR_vars: print("   ", v.name)
        print("🔍 Recon_net 参数列表:")
        for v in Recon_vars: print("   ", v.name)

    def _train(self, begin_epoch):
        tl.files.exists_or_mkdir(test_saving_dir)
        tl.files.exists_or_mkdir(checkpoint_dir)
        tl.files.exists_or_mkdir(log_dir)
        tl.files.exists_or_mkdir(plot_test_loss_dir)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.assign(self.learning_rate, lr_init))
        # 🔥【预热逻辑】首次执行一次图，避免cuDNN初始化卡顿
        print("🔥 正在执行预热操作...")

        dummy_lf = np.zeros([batch_size, img_size, img_size, n_num], dtype=np.float32)
        dummy_syn = np.zeros([batch_size, img_size, img_size, n_num], dtype=np.float32)
        dummy_stack = np.zeros([batch_size, img_size * ReScale_factor[0], img_size * ReScale_factor[1], n_slices], dtype=np.float32)

        feed_warmup = {
            self.plchdr_lf: dummy_lf,
            self.plchdr_SynView: dummy_syn,
            self.plchdr_target3d: dummy_stack,
        }

        _ = self.sess.run(self.losses, feed_dict=feed_warmup)
        print("✅ 预热完成，正式进入训练阶段。")
        epoch_total_loss = 0
        epoch_step_count = 0
        # 🔁 加载断点Checkpoint（根据begin_epoch）
        ckpt_tag = 'epoch{}'.format(begin_epoch)
        sr_ckpt = os.path.join(checkpoint_dir, f'SR_net_{ckpt_tag}.npz')
        recon_ckpt = os.path.join(checkpoint_dir, f'recon_net_{ckpt_tag}.npz')

        if os.path.exists(sr_ckpt):
            tl.files.load_and_assign_npz(sess=self.sess, name=sr_ckpt, network=self.SR_net)
            print(f"🔁 加载 SR checkpoint: {sr_ckpt}")
        if os.path.exists(recon_ckpt):
            tl.files.load_and_assign_npz(sess=self.sess, name=recon_ckpt, network=self.Recon_net)
            print(f"🔁 加载 Recon checkpoint: {recon_ckpt}")
    
        if save_hyperparameters:
            self._save_training_settings()

        dataset_size = self.dataset.prepare(batch_size, n_epoch)
        final_cursor = (dataset_size // batch_size - 1) * batch_size

        while self.dataset.hasNext():
            Stack_batch, HR_batch, LF_batch, cursor, epoch = self.dataset.iter()
            epoch += begin_epoch
            step_start_time = time.time()

            if epoch != 0 and (epoch % decay_every == 0) and cursor == 0:
                new_lr_decay = lr_decay ** (epoch // decay_every)
                self.sess.run(tf.assign(self.learning_rate, lr_init * new_lr_decay))
                print(f"\n🔁 学习率衰减到: {lr_init * new_lr_decay:.8f}")

            feed_train = {
                self.plchdr_target3d: Stack_batch,
                self.plchdr_SynView: HR_batch,
                self.plchdr_lf: LF_batch,
            }

            evaluated = self.sess.run({**self.losses, 'opti_sr': self.SR_optim, 'opti_recon': self.Recon_optim, 'batch_summary': self.merge_op}, feed_train)
            loss_str = [name + ":" + f"{value:.6f}" for name, value in evaluated.items() if 'loss' in name]
            epoch_total_loss += evaluated['total_loss']
            epoch_step_count += 1
            print(f"\rEpoch:[{epoch}/{n_epoch+begin_epoch}] iter:[{cursor}/{dataset_size}] time: {time.time() - step_start_time:.3f}s {time.strftime('%Y-%m-%d %H:%M:%S')} --- {loss_str}", end='')
            
            # 检查NaN并尝试恢复
            has_nan = False
            for name, value in evaluated.items():
                if 'loss' in name and (np.isnan(value) or np.isinf(value)):
                    print(f"\n⚠️ 检测到NaN/Inf损失: {name}={value}! Epoch:{epoch}, Iter:{cursor}")
                    has_nan = True
                    break

            if has_nan:
                print("⚠️ 尝试从上一个检查点恢复...")
                # 恢复到最近的检查点
                last_good_epoch = (epoch // config.TRAIN.ckpt_saving_interval) * config.TRAIN.ckpt_saving_interval
                if last_good_epoch == epoch:  # 如果当前epoch是保存点，回退一个间隔
                    last_good_epoch = max(0, last_good_epoch - config.TRAIN.ckpt_saving_interval)

                # 恢复代码...
                last_sr_ckpt = os.path.join(checkpoint_dir, f'SR_net_epoch{last_good_epoch}.npz')
                last_recon_ckpt = os.path.join(checkpoint_dir, f'recon_net_epoch{last_good_epoch}.npz')

                if os.path.exists(last_sr_ckpt) and os.path.exists(last_recon_ckpt):
                    tl.files.load_and_assign_npz(sess=self.sess, name=last_sr_ckpt, network=self.SR_net)
                    tl.files.load_and_assign_npz(sess=self.sess, name=last_recon_ckpt, network=self.Recon_net)
                    print(f"✅ 成功恢复到epoch {last_good_epoch}")
                else:
                    # 如果没有找到检查点，检查是否有best模型
                    best_sr_ckpt = os.path.join(checkpoint_dir, 'SR_net_best.npz')
                    best_recon_ckpt = os.path.join(checkpoint_dir, 'recon_net_best.npz')

                    if os.path.exists(best_sr_ckpt) and os.path.exists(best_recon_ckpt):
                        tl.files.load_and_assign_npz(sess=self.sess, name=best_sr_ckpt, network=self.SR_net)
                        tl.files.load_and_assign_npz(sess=self.sess, name=best_recon_ckpt, network=self.Recon_net)
                        print("✅ 已恢复到best模型")
                    else:
                        print("❌ 无法找到可恢复的检查点，继续训练但结果可能不稳定")
            self.summary_writer.add_summary(evaluated['batch_summary'], epoch * (dataset_size // batch_size - 1) + cursor // batch_size)

            if cursor == final_cursor:
                if (epoch % config.TRAIN.ckpt_saving_interval == 0):
                    self._save_intermediate_ckpt(epoch, self.sess)
                    self._plot_test_loss()
                with open(os.path.join(checkpoint_dir, "last_epoch.txt"), "w") as f:
                    f.write(str(epoch + 1))
                avg_loss = epoch_total_loss / epoch_step_count
                print(f"\n📉 Epoch {epoch} 平均Loss: {avg_loss:.6f}")
                # 重置统计器
                epoch_total_loss = 0
                epoch_step_count = 0
                if not hasattr(self, 'best_loss') or avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self._save_intermediate_ckpt('best', self.sess)
                    print(f"🌟 当前最优，保存best模型！(epoch {epoch}, avg_loss={avg_loss:.6f})")
                    
        print("\n✅ 训练完成！")

    def _save_training_settings(self):
        params = config.copy()
        if 'load' in params: del params['load']
        save_path = os.path.join(checkpoint_dir, 'train_setting.json')
        with open(save_path, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"📄 训练超参数保存到: {save_path}")

    def _save_intermediate_ckpt(self, tag, sess):
        tag = ('epoch%d' % tag) if is_number(tag) else tag
        sr_file_name = os.path.join(checkpoint_dir, f'SR_net_{tag}.npz')
        recon_file_name = os.path.join(checkpoint_dir, f'recon_net_{tag}.npz')

        # ✅ 彻底替代 all_params，使用名字过滤方式提取 SR/Recon 网络变量
        sr_vars = tl.layers.get_variables_with_name(config.net_setting.SR_model, train_only=True, printable=False)
        recon_vars = tl.layers.get_variables_with_name(config.net_setting.Recon_model, train_only=True, printable=False)

        # ✅ 保存变量
        # ✅ 保存 SR_net 权重（结构化，变量名做 key）
        sr_params_dict = {var.name: sess.run(var) for var in sr_vars}
        np.savez(sr_file_name.replace(".npz", "_structured.npz"), **sr_params_dict)

        # ✅ 保存 Recon_net 权重（结构化，变量名做 key）
        recon_params_dict = {var.name: sess.run(var) for var in recon_vars}
        np.savez(recon_file_name.replace(".npz", "_structured.npz"), **recon_params_dict)
        print(f"💾 保存SR_net:{sr_file_name}, Recon_net:{recon_file_name}")
        with open(os.path.join(checkpoint_dir, 'SR_varnames.txt'), 'w') as f:
            f.write('\n'.join([var.name for var in sr_vars]))
        with open(os.path.join(checkpoint_dir, 'Recon_varnames.txt'), 'w') as f:
            f.write('\n'.join([var.name for var in recon_vars]))


        # 补充推理保存SR和Recon结果
        if hasattr(self, 'test_LFP'):
            test_batch = self.test_LFP[:batch_size]
            SR_view = self.sess.run(self.SR_net.outputs, {self.plchdr_lf: test_batch})
            Recon_stack = self.sess.run(self.Recon_net.outputs, {self.plchdr_lf: test_batch})
            for i in range(SR_view.shape[0]):
                write3d(SR_view[i:i+1], os.path.join(test_saving_dir, f'SR_{tag}_{i}.tif'))
                write3d(Recon_stack[i:i+1], os.path.join(test_saving_dir, f'Recon_{tag}_{i}.tif'))
            print(f"📤 当前epoch {tag} 保存推理SR与Recon图像")



    def _plot_test_loss(self):
        if len(self.test_loss_plt) == 0:
            return
        loss = np.asarray(self.test_loss_plt)
        plt.figure()
        plt.plot(loss[:, 0], loss[:, 1], marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.grid(True)
        save_path = os.path.join(plot_test_loss_dir, 'test_loss.png')
        plt.savefig(save_path)
        np.save(os.path.join(plot_test_loss_dir, 'test_loss.npy'), np.array(self.test_loss_plt))
        plt.close()
        print(f"✅ 验证集loss曲线已保存: {save_path}")

    def train(self, **kwargs):
        try:
            self._train(**kwargs)
        finally:
            self._plot_test_loss()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=int, default=0)
    args = parser.parse_args()

    resume_path = os.path.join(config.TRAIN.ckpt_dir, "last_epoch.txt")
    last_epoch = 0
    if os.path.exists(resume_path):
        with open(resume_path, "r") as f:
            content = f.read().strip()
            if content:
                last_epoch = int(content)

    begin_epoch = args.ckpt if args.ckpt > 0 else last_epoch

    dataset = Dataset(
        config.img_setting.Target3D,
        config.img_setting.Synth_view,
        config.img_setting.LFP,
        config.img_setting.Nnum,
        config.img_setting.img_size // config.img_setting.Nnum,
        n_slices=config.img_setting.n_slices,
        shuffle_for_epoch=shuffle_for_epoch,
        sample_ratio=sample_ratio
    )

    trainer = Trainer(dataset)
    trainer.build_graph()
    trainer.train(begin_epoch=begin_epoch)
